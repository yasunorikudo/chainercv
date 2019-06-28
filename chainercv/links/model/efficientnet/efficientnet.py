from __future__ import division

import math
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import initializers

from chainercv.links import PickableSequentialChain
from chainercv.links.model.efficientnet import utils
from chainercv.links.model.mobilenet import TFConv2DBNActiv
from chainercv.links.model.mobilenet import TFConvolution2D
from chainercv.utils import prepare_pretrained_model

# RGB order
_imagenet_mean = np.array(
    [123.675, 116.28, 103.53],
    dtype=np.float32)[:, np.newaxis, np.newaxis]
_imagenet_scale = 1. / np.array(
    [58.395, 57.120003, 57.375],
    dtype=np.float32)[:, np.newaxis, np.newaxis]
_bn_kwargs = {
    "decay": 0.99,
    "eps": 0.001,
    "dtype": chainer.config.dtype
}

def swish(x):
    return x * F.sigmoid(x)


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
                 drop_connect_rate=None, bn_kwargs={}):
        super(MBConvBlock, self).__init__()
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.has_se = (se_ratio is not None) \
            and (se_ratio > 0) and (se_ratio <= 1)
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
                self.bn = L.BatchNormalization(middle_filters, **bn_kwargs)

            self.depthwise_conv2d = TFConvolution2D(
                in_channels=middle_filters,
                out_channels=middle_filters,
                ksize=kernel_size,
                stride=strides,
                pad='SAME',
                nobias=True,
                groups=middle_filters)
            bn_name = 'bn' if expand_ratio == 1 else 'bn_1'
            setattr(self, bn_name,
                    L.BatchNormalization(middle_filters, **bn_kwargs))

            if self.has_se:
                self.se = SEBlock(
                    middle_filters, int(input_filters * se_ratio))

            conv_name = 'conv2d' if expand_ratio == 1 else 'conv2d_1'
            setattr(self, conv_name,
                    TFConvolution2D(
                        in_channels=middle_filters,
                        out_channels=output_filters,
                        ksize=1,
                        pad='SAME',
                        nobias=True))
            bn_name = 'bn_1' if expand_ratio == 1 else 'bn_2'
            setattr(self, bn_name,
                    L.BatchNormalization(output_filters, **bn_kwargs))


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
            if (np.array(self.strides) == 1).all() \
                    and self.input_filters == self.output_filters:
                h = h + x
        return h


class MBConvs(chainer.Sequential):

    def __init__(self, block_args, efficientnet_params, bn_kwargs):
        block_args = block_args.copy()
        width_coefficient, depth_coefficient = efficientnet_params[:2]

        block_args['num_repeat'] = int(
            math.ceil(depth_coefficient * block_args['num_repeat']))
        block_args['input_filters'] = utils.round_filters(
            block_args['input_filters'], width_coefficient)
        block_args['output_filters'] = utils.round_filters(
            block_args['output_filters'], width_coefficient)

        blocks = []
        for i in range(block_args['num_repeat']):
            blocks.append(MBConvBlock(**block_args, bn_kwargs=bn_kwargs))
            block_args['input_filters'] = block_args['output_filters']
            block_args['strides'] = [1, 1]

        super(MBConvs, self).__init__(*blocks)


class EfficientNet(PickableSequentialChain):

    _models = {
        'efficientnet-b0': {
            'imagenet': {
                'param': {'n_class': 1000, 'mean': _imagenet_mean,
                          'scale': _imagenet_scale, 'bn_kwargs': _bn_kwargs},
                'overwritable': {'mean', 'scale', 'bn_kwargs'},
                'url': 'https://www.dropbox.com/s/6ahq41qlomfpahw/'
                'efficientnet-b0_imagenet_converted_2019_06_28.npz'
            },
        },
        'efficientnet-b1': {
            'imagenet': {
                'param': {'n_class': 1000, 'mean': _imagenet_mean,
                          'scale': _imagenet_scale, 'bn_kwargs': _bn_kwargs},
                'overwritable': {'mean', 'scale', 'bn_kwargs'},
                'url': 'https://www.dropbox.com/s/wdvey1o23s5190o/'
                'efficientnet-b1_imagenet_converted_2019_06_28.npz'
            },
        },
        'efficientnet-b2': {
            'imagenet': {
                'param': {'n_class': 1000, 'mean': _imagenet_mean,
                          'scale': _imagenet_scale, 'bn_kwargs': _bn_kwargs},
                'overwritable': {'mean', 'scale', 'bn_kwargs'},
                'url': 'https://www.dropbox.com/s/9ix5qyagtn34tig/'
                'efficientnet-b2_imagenet_converted_2019_06_28.npz'
            },
        },
        'efficientnet-b3': {
            'imagenet': {
                'param': {'n_class': 1000, 'mean': _imagenet_mean,
                          'scale': _imagenet_scale, 'bn_kwargs': _bn_kwargs},
                'overwritable': {'mean', 'scale', 'bn_kwargs'},
                'url': 'https://www.dropbox.com/s/a4dcwfaphc4q1ml/'
                'efficientnet-b3_imagenet_converted_2019_06_28.npz'
            },
        },
        'efficientnet-b4': {
            'imagenet': {
                'param': {'n_class': 1000, 'mean': _imagenet_mean,
                          'scale': _imagenet_scale, 'bn_kwargs': _bn_kwargs},
                'overwritable': {'mean', 'scale', 'bn_kwargs'},
                'url': 'https://www.dropbox.com/s/t10f1clatibk666/'
                'efficientnet-b4_imagenet_converted_2019_06_28.npz'
            },
        },
        'efficientnet-b5': {
            'imagenet': {
                'param': {'n_class': 1000, 'mean': _imagenet_mean,
                          'scale': _imagenet_scale, 'bn_kwargs': _bn_kwargs},
                'overwritable': {'mean', 'scale', 'bn_kwargs'},
                'url': 'https://www.dropbox.com/s/djqwmf9wsi32ldc/'
                'efficientnet-b5_imagenet_converted_2019_06_28.npz'
            },
        },
    }

    def __init__(self, model_name='efficientnet-b0',
                 n_class=None,
                 pretrained_model=None,
                 mean=None, scale=None, initialW=None,
                 bn_kwargs=None, fc_kwargs={}):

        param, path = prepare_pretrained_model(
            {'n_class': n_class, 'mean': mean,
             'scale': scale, 'bn_kwargs': bn_kwargs},
            pretrained_model, self._models[model_name],
            {'n_class': 1000, 'mean': _imagenet_mean,
             'scale': _imagenet_scale, 'bn_kwargs': {}})
        self.mean = param['mean']
        self.scale = param['scale']
        bn_kwargs = param['bn_kwargs']

        if initialW is None:
            initialW = initializers.HeNormal(scale=1., fan_option='fan_out')
        if 'initialW' not in fc_kwargs:
            fc_kwargs['initialW'] = initializers.Normal(scale=0.01)
        if pretrained_model:
            # As a sampling process is time-consuming,
            # we employ a zero initializer for faster computation.
            initialW = initializers.constant.Zero()
            fc_kwargs['initialW'] = initializers.constant.Zero()

        super(EfficientNet, self).__init__()
        self.model_name = model_name
        block_args = utils.get_block_args()
        efficientnet_params = utils.get_efficientnet_params(model_name)
        width_coefficient, _, self.insize, _ = efficientnet_params
        with self.init_scope():
            self.conv0 = TFConv2DBNActiv(
                3,
                utils.round_filters(32, width_coefficient),
                ksize=3,
                stride=2,
                pad='SAME',
                nobias=True,
                activ=swish,
                bn_kwargs=bn_kwargs)
            for i, a in enumerate(block_args):
                setattr(self, 'block{}'.format(i + 1),
                        MBConvs(a, efficientnet_params, bn_kwargs))
            self.conv8 = TFConv2DBNActiv(
                None,
                utils.round_filters(1280, width_coefficient),
                ksize=1,
                stride=1,
                pad='SAME',
                nobias=True,
                activ=swish,
                bn_kwargs=bn_kwargs)
            self.pool9 = lambda x: F.average(x, axis=(2, 3))
            self.fc10 = L.Linear(None, param['n_class'], **fc_kwargs)

        if path:
            chainer.serializers.load_npz(path, self)


class EfficientNetB0(EfficientNet):

    def __init__(self, n_class=None,
                 pretrained_model=None,
                 mean=None, scale=None, initialW=None,
                 bn_kwargs=None, fc_kwargs={}):
        super(EfficientNetB0, self).__init__(
            'efficientnet-b0', n_class, pretrained_model,
            mean, scale, initialW, bn_kwargs, fc_kwargs)



class EfficientNetB1(EfficientNet):

    def __init__(self, n_class=None,
                 pretrained_model=None,
                 mean=None, scale=None, initialW=None,
                 bn_kwargs=None, fc_kwargs={}):
        super(EfficientNetB1, self).__init__(
            'efficientnet-b1', n_class, pretrained_model,
            mean, scale, initialW, bn_kwargs, fc_kwargs)


class EfficientNetB2(EfficientNet):

    def __init__(self, n_class=None,
                 pretrained_model=None,
                 mean=None, scale=None, initialW=None,
                 bn_kwargs=None, fc_kwargs={}):
        super(EfficientNetB2, self).__init__(
            'efficientnet-b2', n_class, pretrained_model,
            mean, scale, initialW, bn_kwargs, fc_kwargs)


class EfficientNetB3(EfficientNet):

    def __init__(self, n_class=None,
                 pretrained_model=None,
                 mean=None, scale=None, initialW=None,
                 bn_kwargs=None, fc_kwargs={}):
        super(EfficientNetB3, self).__init__(
            'efficientnet-b3', n_class, pretrained_model,
            mean, scale, initialW, bn_kwargs, fc_kwargs)


class EfficientNetB4(EfficientNet):

    def __init__(self, n_class=None,
                 pretrained_model=None,
                 mean=None, scale=None, initialW=None,
                 bn_kwargs=None, fc_kwargs={}):
        super(EfficientNetB4, self).__init__(
            'efficientnet-b4', n_class, pretrained_model,
            mean, scale, initialW, bn_kwargs, fc_kwargs)


class EfficientNetB5(EfficientNet):

    def __init__(self, n_class=None,
                 pretrained_model=None,
                 mean=None, scale=None, initialW=None,
                 bn_kwargs=None, fc_kwargs={}):
        super(EfficientNetB5, self).__init__(
            'efficientnet-b5', n_class, pretrained_model,
            mean, scale, initialW, bn_kwargs, fc_kwargs)
