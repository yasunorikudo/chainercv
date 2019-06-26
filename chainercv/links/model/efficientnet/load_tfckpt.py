import argparse
import os
from tensorflow.python import pywrap_tensorflow
import numpy as np

import chainer
from chainer import cuda

from chainercv.links.model.efficientnet.efficientnet import EfficientNet
from chainercv.links.model.efficientnet.efficientnet import get_efficientnet_params
from chainercv.links.model.efficientnet.efficientnet import get_block_args

def get_ch2tf_dict(model):
    ch2tf_dict = {
        '/conv0/conv/conv/W': '{}/stem/conv2d/kernel'.format(model.model_name),
        '/conv0/bn/beta': '{}/stem/tpu_batch_normalization/beta'.format(model.model_name),
        '/conv0/bn/gamma': '{}/stem/tpu_batch_normalization/gamma'.format(model.model_name),
        '/conv8/conv/conv/W': '{}/head/conv2d/kernel'.format(model.model_name),
        '/conv8/bn/beta': '{}/head/tpu_batch_normalization/beta'.format(model.model_name),
        '/conv8/bn/gamma': '{}/head/tpu_batch_normalization/gamma'.format(model.model_name),
        '/fc10/W': '{}/head/dense/kernel'.format(model.model_name),
        '/fc10/b': '{}/head/dense/bias'.format(model.model_name),
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
                model.model_name, n_tf_blocks[key])

            if child_names[2].startswith('bn'):
                child_names[2] = child_names[2].replace('bn', 'tpu_batch_normalization')

            for child_name in child_names[2:]:
                tf_param_name += '/{}'.format(child_name)

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
    model(x)

    ch2tf_dict = get_ch2tf_dict(model)
    reader = pywrap_tensorflow.NewCheckpointReader(args.filename)
    for name, param in model.namedparams():
        tf_param_name = ch2tf_dict[name]
        arr = reader.get_tensor(tf_param_name)
        if len(param.shape) == 1:
            param.data[:] = arr
        elif len(param.shape) == 2:
            param.data[:] = arr.transpose(1, 0)
        elif len(param.shape) == 4:
            if 'depthwise' in name:
                param.data[:] = arr.transpose(2, 3, 0, 1)
            else:
                param.data[:] = arr.transpose(3, 2, 0, 1)
        else:
            raise Exception()

    if args.out:
        chainer.save_npz(args.out, model)
