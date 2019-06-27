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
from chainercv.links.model.mobilenet.tf_conv_2d_bn_activ import TFConv2DBNActiv
from chainercv.links.model.mobilenet.tf_convolution_2d import TFConvolution2D

BN_EPS = 1e-3
BN_DECAY = 0.99

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


class SEBlock(chainer.Chain):

    def __init__(self, in_channels, mid_channels):
        super(SEBlock, self).__init__()
        with self.init_scope():
            self.conv2d = TFConvolution2D(
                in_channels=in_channels,
                out_channels=mid_channels,
                ksize=1,
                stride=1,
                pad='SAME')
            self.conv2d_1 = TFConvolution2D(
                in_channels=mid_channels,
                out_channels=in_channels,
                ksize=1,
                stride=1,
                pad='SAME')

    def __call__(self, x):
        se = F.average_pooling_2d(x, ksize=x.shape[2:])
        se = swish(self.conv2d(se))
        se = self.conv2d_1(se)
        h = x * F.sigmoid(se)
        return h


class MBConvBlock(chainer.Chain):

    def __init__(self, kernel_size, num_repeat, input_filters, output_filters,
                 expand_ratio, id_skip, se_ratio, strides,
                 drop_connect_rate=None):
        super(MBConvBlock, self).__init__()
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.has_se = (se_ratio is not None) and (se_ratio > 0) and (se_ratio <= 1)
        self.expand_ratio = expand_ratio
        self.id_skip = id_skip
        self.strides = strides

        with self.init_scope():
            middle_filters = int(expand_ratio * input_filters)
            if expand_ratio != 1:
                self.conv2d = TFConvolution2D(
                    in_channels=input_filters,
                    out_channels=middle_filters,
                    ksize=1,
                    pad='SAME',
                    nobias=True)
                self.bn = L.BatchNormalization(middle_filters, decay=BN_DECAY, eps=BN_EPS)

            self.depthwise_conv2d = TFConvolution2D(
                in_channels=middle_filters,
                out_channels=middle_filters,
                ksize=kernel_size,
                stride=strides,
                pad='SAME',
                nobias=True,
                groups=middle_filters)
            bn_name = 'bn' if expand_ratio == 1 else 'bn_1'
            setattr(self, bn_name, L.BatchNormalization(middle_filters, decay=BN_DECAY, eps=BN_EPS))

            if self.has_se:
                self.se = SEBlock(middle_filters, int(input_filters * se_ratio))

            conv_name = 'conv2d' if expand_ratio == 1 else 'conv2d_1'
            setattr(self, conv_name,
                    TFConvolution2D(
                        in_channels=middle_filters,
                        out_channels=output_filters,
                        ksize=1,
                        pad='SAME',
                        nobias=True))
            bn_name = 'bn_1' if expand_ratio == 1 else 'bn_2'
            setattr(self, bn_name, L.BatchNormalization(output_filters, decay=BN_DECAY, eps=BN_EPS))


    def __call__(self, x):
        if self.expand_ratio != 1:
            h = swish(self.bn(self.conv2d(x)))
        else:
            h = x
        bn_name = 'bn' if self.expand_ratio == 1 else 'bn_1'
        h = swish(self[bn_name](self.depthwise_conv2d(h)))
        if self.has_se:
            h = self.se(h)

        conv_name = 'conv2d' if self.expand_ratio == 1 else 'conv2d_1'
        bn_name = 'bn_1' if self.expand_ratio == 1 else 'bn_2'
        h = self[bn_name](self[conv_name](h))

        if self.id_skip:
            if (np.array(self.strides) == 1).all() and self.input_filters == self.output_filters:
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
        for i in range(block_args['num_repeat']):
            blocks.append(MBConvBlock(**block_args))
            block_args['input_filters'] = block_args['output_filters']
            block_args['strides'] = [1, 1]

        super(MBConvs, self).__init__(*blocks)


class EfficientNet(PickableSequentialChain):

    def __init__(self, model_name='efficientnet-b0'):
        super(EfficientNet, self).__init__()
        self.model_name = model_name
        block_args = get_block_args()
        efficientnet_params = get_efficientnet_params(model_name)
        width_coefficient = efficientnet_params[0]
        with self.init_scope():
            self.conv0 = TFConv2DBNActiv(
                3,
                round_filters(32, width_coefficient),
                ksize=3,
                stride=2,
                pad='SAME',
                nobias=True,
                activ=swish,
                bn_kwargs={'decay': BN_DECAY, 'eps': BN_EPS})
            for i, a in enumerate(block_args):
                setattr(self, 'block{}'.format(i + 1), MBConvs(a, efficientnet_params))
            self.conv8 = TFConv2DBNActiv(
                None,
                round_filters(1280, width_coefficient),
                ksize=1,
                stride=1,
                pad='SAME',
                nobias=True,
                activ=swish,
                bn_kwargs={'decay': BN_DECAY, 'eps': BN_EPS})
            self.pool9 = lambda x: F.average(x, axis=(2, 3))
            self.fc10 = L.Linear(None, 1000)
