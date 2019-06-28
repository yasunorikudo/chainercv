# EfficientNet

For evaluation, please go to [`examples/classification`](https://github.com/chainer/chainercv/tree/master/examples/classification).

## Convert TensorFlow model
Convert TensorFlow's `*.ckpt` to `*.npz`.

```
$ python tfckpt2npz.py efficientnet-b0 <source>.ckpt <target>.npz
```

The pretrained `.ckpt` for efficientnet can be downloaded from here.
https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
