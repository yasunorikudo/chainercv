from __future__ import division

import collections
import math
import numpy as np
import re

import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L

from chainercv.links import Conv2DBNActiv
from chainercv.links import PickableSequentialChain
from chainercv import utils


# Ref. https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_builder.py
def get_efficientnet_params(model_name):
    """Get efficientnet params based on model name."""
    params_dict = {
        # (width_coefficient, depth_coefficient, resolution, dropout_rate)
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    }
    return params_dict[model_name]

def decode_block_string(block_string):
    """Gets a block through a string notation of arguments."""
    assert isinstance(block_string, str)
    ops = block_string.split('_')
    options = {}
    for op in ops:
        splits = re.split(r'(\d.*)', op)
        if len(splits) >= 2:
            key, value = splits[:2]
            options[key] = value

    if 's' not in options or len(options['s']) != 2:
        raise ValueError('Strides options should be a pair of integers.')

    return collections.OrderedDict(
        kernel_size=int(options['k']),
        num_repeat=int(options['r']),
        input_filters=int(options['i']),
        output_filters=int(options['o']),
        expand_ratio=int(options['e']),
        id_skip=('noskip' not in block_string),
        se_ratio=float(options['se']) if 'se' in options else None,
        strides=[int(options['s'][0]), int(options['s'][1])])

def get_block_args():
    blocks_args = [
    'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
    'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
    'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
    'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    return [decode_block_string(block_string) for block_string in blocks_args]

def swish(x):
    return x * F.sigmoid(x)

def round_filters(filters, width_coefficient, depth_divisor=8, min_depth=None):
    """Round number of filters based on depth multiplier."""
    orig_f = filters
    multiplier = width_coefficient
    divisor = depth_divisor
    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


class MBConvBlock(chainer.Chain):

    def __init__(self, kernel_size, num_repeat, input_filters, output_filters, expand_ratio, id_skip, se_ratio, strides, drop_connect_rate=None):
        super(MBConvBlock, self).__init__()

        self.input_filters = input_filters
        self.output_filters = output_filters
        self.has_se = (se_ratio is not None) and (se_ratio > 0) and (se_ratio <= 1)
        self.expand_ratio = expand_ratio
        self.id_skip = id_skip
        self.strides = strides

        with self.init_scope():
            if expand_ratio != 1:
                self.expand_conv = L.Convolution2D(
                    in_channels=input_filters,
                    out_channels=int(expand_ratio * input_filters),
                    ksize=1,
                    nobias=True)
                self.bn0 = L.BatchNormalization(int(expand_ratio * input_filters))

            self.depthwise_conv = L.DepthwiseConvolution2D(
                in_channels=int(expand_ratio * input_filters),
                channel_multiplier=1,
                ksize=kernel_size,
                stride=strides,
                pad=int((kernel_size - 1) / 2),
                nobias=True)
            self.bn1 = L.BatchNormalization(int(expand_ratio * input_filters))

            if self.has_se:
                self.se_recude = L.Linear(int(expand_ratio * input_filters), int(input_filters * se_ratio))
                self.se_expand = L.Linear(int(input_filters * se_ratio), int(expand_ratio * input_filters))

            self.project_conv = L.Convolution2D(
                in_channels=int(expand_ratio * input_filters),
                out_channels=output_filters,
                ksize=1,
                nobias=True)
            self.bn2 = L.BatchNormalization(output_filters)

    def __call__(self, x):
        if self.expand_ratio != 1:
            h = swish(self.b0(self.expand_conv(x)))
        else:
            h = x
        h = swish(self.bn1(self.depthwise_conv(h)))
        if self.has_se:
            se = F.average_pooling_2d(h, ksize=h.shape[2:])
            se = self.se_expand(swish(self.se_recude(se)))
            h = h * F.sigmoid(se)[:, :, None, None]
        h = self.bn2(self.project_conv(h))
        if self.id_skip:
            if all(self.strides) and self.input_filters == self.output_filters:
                h = h + x
        return h


class MBConvs(chainer.Sequential):

    def __init__(self, block_args, efficientnet_params):
        block_args = block_args.copy()
        width_coefficient, depth_coefficient = efficientnet_params[:2]

        block_args['num_repeat'] = int(math.ceil(depth_coefficient * block_args['num_repeat']))
        block_args['input_filters'] = round_filters(block_args['input_filters'], width_coefficient)
        block_args['output_filters'] = round_filters(block_args['output_filters'], width_coefficient)

        blocks = []
        for i in block_args['num_repeat']:
            blocks.append(MBConvBlock(**block_args))
            block_args['input_filters'] = block_args['output_filters']







# class EfficientNet(PickableSequentialChain):
#
#     def __init__(self, width_coefficient, depth_coefficient, resolution, dropout_rate, block_args):
#         super(EfficientNet, self).__init__()



if __name__ == '__main__':
    # model = EfficientNet()
    x = np.random.randn(1, 32, 12, 12).astype('f')
    aaa = get_block_args()
    model = MBConvs(**aaa[0])
    y = model(x)
    import ipdb; ipdb.set_trace()
