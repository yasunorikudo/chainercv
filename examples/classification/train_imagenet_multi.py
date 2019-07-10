from __future__ import division
import argparse
import multiprocessing
import numpy as np
import os
import yaml

import chainer
from chainer import iterators
from chainer.links import Classifier
from chainer.optimizers import CorrectedMomentumSGD
from chainer import training
from chainer.training import extensions

import chainercv
from chainercv.chainer_experimental.datasets.sliceable import TransformDataset
from chainercv.datasets import directory_parsing_label_names
from chainercv.datasets import DirectoryParsingLabelDataset
from chainercv.transforms import center_crop
from chainercv.transforms import random_flip
from chainercv.transforms import random_sized_crop
from chainercv.transforms import resize
from chainercv.transforms import scale

from chainercv.chainer_experimental.training.extensions import make_shift
from chainercv.chainer_experimental.optimizers import MomentumRMSprop

from chainercv.links.model.resnet import Bottleneck

import chainermn

# https://docs.chainer.org/en/stable/tips.html#my-training-process-gets-stuck-when-using-multiprocessiterator
try:
    import cv2
    cv2.setNumThreads(0)
except ImportError:
    pass


class TrainTransform(object):

    def __init__(self, mean, scale, insize):
        self.mean = mean
        self.scale = scale
        self.insize = insize

    def __call__(self, in_data):
        img, label = in_data
        img = random_sized_crop(img)
        img = resize(img, (self.insize, self.insize))
        img = random_flip(img, x_random=True)
        img -= self.mean
        img *= self.scale
        return img, label


class ValTransform(object):

    def __init__(self, mean, scale, insize):
        self.mean = mean
        self.scale = scale
        self.insize = insize

    def __call__(self, in_data):
        img, label = in_data
        img = scale(img, self.insize + 32)
        img = center_crop(img, (self.insize, self.insize))
        img -= self.mean
        img *= self.scale
        return img, label


def main():
    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument('train', help='Path to root of the train dataset')
    parser.add_argument('val', help='Path to root of the validation dataset')
    parser.add_argument('--config',
                        '-c', type=str, default='configs/resnet50.yaml')
    parser.add_argument('--communicator', type=str,
                        default='pure_nccl', help='Type of communicator')
    parser.add_argument('--loaderjob', type=int, default=4)
    parser.add_argument('--out', type=str, default='result')
    parser.add_argument('--resume', type=str, default='')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    model_cfg = cfg['model']
    train_cfg = cfg['training']

    # https://docs.chainer.org/en/stable/chainermn/tutorial/tips_faqs.html#using-multiprocessiterator
    if hasattr(multiprocessing, 'set_start_method'):
        multiprocessing.set_start_method('forkserver')
        p = multiprocessing.Process()
        p.start()
        p.join()

    comm = chainermn.create_communicator(args.communicator)
    device = comm.intra_rank

    lr = train_cfg['lr_scale'] * (train_cfg['batchsize'] * comm.size) / 256
    if comm.rank == 0:
        print('lr={}: lr is selected based on the linear '
              'scaling rule'.format(lr))

    label_names = directory_parsing_label_names(args.train)


    extractor = getattr(chainercv.links, model_cfg['class'])(
        n_class=len(label_names), **model_cfg['kwargs'])
    extractor.pick = model_cfg['score_layer_name']
    model = Classifier(extractor)
    # Following https://arxiv.org/pdf/1706.02677.pdf,
    # the gamma of the last BN of each resblock is initialized by zeros.
    if model_cfg['class'] in ['ResNet50', 'ResNet101', 'ResNet152']:
        for l in model.links():
            if isinstance(l, Bottleneck):
                l.conv3.bn.gamma.data[:] = 0

    train_data = DirectoryParsingLabelDataset(args.train)
    val_data = DirectoryParsingLabelDataset(args.val)

    mean = extractor.mean if hasattr(extractor, 'mean') else 0
    scale = extractor.scale if hasattr(extractor, 'scale') else 1
    insize = model_cfg['insize']
    train_data = TransformDataset(
        train_data, ('img', 'label'), TrainTransform(mean, scale, insize))
    val_data = TransformDataset(
        val_data, ('img', 'label'), ValTransform(mean, scale, insize))
    print('finished loading dataset')

    if comm.rank == 0:
        train_indices = np.arange(len(train_data))
        val_indices = np.arange(len(val_data))
    else:
        train_indices = None
        val_indices = None

    train_indices = chainermn.scatter_dataset(
        train_indices, comm, shuffle=True)
    val_indices = chainermn.scatter_dataset(val_indices, comm, shuffle=True)
    train_data = train_data.slice[train_indices]
    val_data = val_data.slice[val_indices]
    train_iter = chainer.iterators.MultiprocessIterator(
        train_data, train_cfg['batchsize'], n_processes=args.loaderjob)
    val_iter = iterators.MultiprocessIterator(
        val_data, train_cfg['batchsize'],
        repeat=False, shuffle=False, n_processes=args.loaderjob)

    optimizers = {'CorrectedMomentumSGD': CorrectedMomentumSGD,
                  'MomentumRMSprop': MomentumRMSprop}
    optimizer_class = optimizers[train_cfg['optimizer']['class']]
    optimizer = chainermn.create_multi_node_optimizer(
        optimizer_class(lr=lr, **train_cfg['optimizer']['kwargs']), comm)
    optimizer.setup(model)
    for hook_cfg in train_cfg['optimizer_hooks']:
        for param in model.params():
            hook = getattr(chainer.optimizer_hooks, hook_cfg['class'])
            if param.name not in hook_cfg['ignore_params']:
                param.update_rule.add_hook(hook(**hook_cfg['kwargs']))

    if device >= 0:
        chainer.cuda.get_device(device).use()
        model.to_gpu()

    updater = chainer.training.StandardUpdater(
        train_iter, optimizer, device=device)

    trainer = training.Trainer(
        updater, (train_cfg['epoch'], 'epoch'), out=args.out)

    @make_shift('lr')
    def warmup_and_exponential_shift(trainer):

        if 'timing' in train_cfg['lr_scheduling'] \
                and 'frequency' in train_cfg['lr_scheduling']:
            raise Exception('')

        epoch = trainer.updater.epoch_detail
        if 'warmup' in train_cfg['lr_scheduling']:
            warmup_epoch = train_cfg['lr_scheduling']['warmup']['epoch']
            warmup_lr = train_cfg['lr_scheduling']['warmup']['lr']
            if epoch < warmup_epoch:
                if lr > warmup_lr:
                    warmup_rate = warmup_lr / lr
                    rate = warmup_rate \
                        + (1 - warmup_rate) * epoch / warmup_epoch
                    return rate * lr
                else:
                    return lr

        if 'timing' in train_cfg['lr_scheduling']['exp_shift']:
            timing = train_cfg['lr_scheduling']['exp_shift']['timing']
            for i, e in enumerate(timing):
                if epoch < e:
                    rate = np.power(
                        train_cfg['lr_scheduling']['exp_shift']['rate'], i)
                    return rate * lr
            rate = np.power(
                train_cfg['lr_scheduling']['exp_shift']['rate'], len(timing))
            return rate * lr

        freq = train_cfg['lr_scheduling']['exp_shift']['frequency']
        rate = np.power(
            train_cfg['lr_scheduling']['exp_shift']['rate'], int(epoch / freq))
        return rate * lr

    trainer.extend(warmup_and_exponential_shift)
    evaluator = chainermn.create_multi_node_evaluator(
        extensions.Evaluator(val_iter, model, device=device), comm)
    trainer.extend(evaluator, trigger=(1, 'epoch'))

    log_interval = 0.1, 'epoch'
    print_interval = 0.1, 'epoch'

    if comm.rank == 0:
        trainer.extend(chainer.training.extensions.observe_lr(),
                       trigger=log_interval)
        trainer.extend(
            extensions.snapshot_object(
                extractor, 'snapshot_model_{.updater.epoch}.npz'),
            trigger=(train_cfg['epoch'], 'epoch'))
        trainer.extend(
            extensions.snapshot(filename='snapshot_latest'),
            trigger=(1, 'epoch'))
        trainer.extend(extensions.LogReport(trigger=log_interval))
        trainer.extend(extensions.PrintReport(
            ['iteration', 'epoch', 'elapsed_time', 'lr',
             'main/loss', 'validation/main/loss',
             'main/accuracy', 'validation/main/accuracy']
        ), trigger=print_interval)
        trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.resume and os.path.exists(args.resume):
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
