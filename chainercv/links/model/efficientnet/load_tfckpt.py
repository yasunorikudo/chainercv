import argparse
import os
from tensorflow.python import pywrap_tensorflow
import numpy as np

import chainer
from chainer import cuda

from chainercv.links.model.efficientnet.efficientnet import EfficientNet
from chainercv.links.model.efficientnet.efficientnet import get_efficientnet_params
from chainercv.links.model.efficientnet.efficientnet import get_block_args

def get_param_from_name(name, model):
    obj = model
    for child_name in name.split('/')[1:]:
        if child_name.isdecimal():
            # See https://github.com/chainer/chainer/issues/6053
            i = 0 if int(child_name) == 0 else len(obj) - int(child_name)
            obj = obj[i]
        else:
            obj = getattr(obj, child_name)

    if isinstance(obj, chainer.variable.Parameter):
        return obj.data
    return obj

def get_tensor(ckpt_reader, name, ema_ratio=0.999):
    if (name + '/ExponentialMovingAverage'
        ) in ckpt_reader.get_variable_to_shape_map().keys():
        base = ckpt_reader.get_tensor(name)
        ema = ckpt_reader.get_tensor(name + '/ExponentialMovingAverage')

        return (1.0 - ema_ratio) * base + ema_ratio * ema
    else:
        return ckpt_reader.get_tensor(name)

def get_ch2tf_dict(model):
    model_name = model.model_name
    ch2tf_dict = {
        '/conv0/conv/conv/W': '{}/stem/conv2d/kernel'.format(model_name),
        '/conv0/bn/beta': '{}/stem/tpu_batch_normalization/beta'.format(model_name),
        '/conv0/bn/gamma': '{}/stem/tpu_batch_normalization/gamma'.format(model_name),
        '/conv0/bn/avg_mean': '{}/stem/tpu_batch_normalization/moving_mean'.format(model_name),
        '/conv0/bn/avg_var': '{}/stem/tpu_batch_normalization/moving_variance'.format(model_name),
        '/conv8/conv/conv/W': '{}/head/conv2d/kernel'.format(model_name),
        '/conv8/bn/beta': '{}/head/tpu_batch_normalization/beta'.format(model_name),
        '/conv8/bn/gamma': '{}/head/tpu_batch_normalization/gamma'.format(model_name),
        '/conv8/bn/avg_mean': '{}/head/tpu_batch_normalization/moving_mean'.format(model_name),
        '/conv8/bn/avg_var': '{}/head/tpu_batch_normalization/moving_variance'.format(model_name),
        '/fc10/W': '{}/head/dense/kernel'.format(model_name),
        '/fc10/b': '{}/head/dense/bias'.format(model_name),
    }
    chainer_param_names = sorted([name for name, _ in model.namedparams()])
    n_tf_blocks = {}
    for name in chainer_param_names:
        child_names = name.split('/')[1:]
        if child_names[0].startswith('block'):
            key = (int(child_names[0][5:]), int(child_names[1]))
            if not key in n_tf_blocks:
                n_tf_blocks[key] = len(n_tf_blocks)

            tf_param_name = '{}/blocks_{}'.format(
                model_name, n_tf_blocks[key])

            if child_names[2].startswith('bn'):
                child_names[2] = child_names[2].replace('bn', 'tpu_batch_normalization')

            for child_name in child_names[2:]:
                tf_param_name += '/{}'.format(child_name)

            if child_names[2].startswith('tpu_batch_normalization') and child_names[-1] == 'gamma':
                tf_param_name_bn_mean = tf_param_name.replace('gamma', 'moving_mean')
                tf_param_name_bn_var = tf_param_name.replace('gamma', 'moving_variance')
                ch2tf_dict[name.replace('gamma', 'avg_mean')] = tf_param_name_bn_mean
                ch2tf_dict[name.replace('gamma', 'avg_var')] = tf_param_name_bn_var

            if child_names[2] == 'depthwise_conv2d':
                tf_param_name = tf_param_name.replace('conv/W', 'depthwise_kernel')
                tf_param_name = tf_param_name.replace('conv/b', 'depthwise_bias')
            else:
                tf_param_name = tf_param_name.replace('conv/W', 'kernel')
                tf_param_name = tf_param_name.replace('conv/b', 'bias')
            ch2tf_dict[name] = tf_param_name

    return ch2tf_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help='Path to model.ckpt')
    parser.add_argument('--model_name', '-m', type=str, default='efficientnet-b0')
    parser.add_argument('--out', '-o', type=str, default=None)
    args = parser.parse_args()

    insize = get_efficientnet_params(args.model_name)[2]
    x = np.zeros((2, 3, insize, insize), dtype=np.float32)
    model = EfficientNet(args.model_name)
    model(x)  # Determine shapes of all params

    ch2tf_dict = get_ch2tf_dict(model)
    reader = pywrap_tensorflow.NewCheckpointReader(args.filename)
    for name in ch2tf_dict.keys():
        tf_param_name = ch2tf_dict[name]
        arr = get_tensor(reader, tf_param_name, ema_ratio=1.0)
        param = get_param_from_name(name, model)
        if len(param.shape) == 1:
            param[:] = arr
        elif len(param.shape) == 2:
            param[:] = arr.transpose(1, 0)
        elif len(param.shape) == 4:
            if 'depthwise' in name:
                param[:] = arr.transpose(2, 3, 0, 1)
            else:
                param[:] = arr.transpose(3, 2, 0, 1)
        else:
            raise Exception()

    if args.out:
        chainer.serializers.save_npz(args.out, model)
