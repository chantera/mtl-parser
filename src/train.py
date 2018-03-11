#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
from teras.app import App, arg
import teras.framework.chainer as framework_utils
import teras.logging as Log
from teras.training import Trainer

import dataset
import models
import transition
import utils


def train(
        train_file,
        test_file=None,
        embed_file=None,
        embed_size=100,
        n_epoch=20,
        batch_size=32,
        lr=0.001,
        l2_lambda=0.0,
        grad_clip=5.0,
        tasks='tp',
        gpu=-1,
        save_to=None,
        seed=None):
    if seed is not None:
        utils.set_random_seed(seed, gpu)
        Log.i("random seed: {}".format(seed))
    framework_utils.set_debug(App.debug)

    # Select Task
    with_tagging_task = False
    with_parsing_task = False
    for char in tasks:
        if char == 't':
            with_tagging_task = True
        elif char == 'p':
            with_parsing_task = True
        else:
            raise ValueError("Invalid task specified: {}".format(char))
    if not any([with_tagging_task, with_parsing_task]):
        raise RuntimeError("No valid task specified")
    Log.i('Task: tagging={}, parsing={}'
          .format(with_tagging_task, with_parsing_task))

    # Transition System
    transition_system = transition.ArcStandard
    if with_parsing_task:
        Log.i('Transition System: {}'.format(transition_system))

    # Load files
    Log.i('initialize DataLoader with embed_file={} and embed_size={}'
          .format(embed_file, embed_size))
    loader = dataset.DataLoader(word_embed_file=embed_file,
                                word_embed_size=embed_size,
                                char_embed_size=10,
                                transition_system=transition_system)
    Log.i('load train dataset from {}'.format(train_file))
    train_dataset = loader.load(train_file, train=True,
                                size=120 if utils.is_dev() else None)
    if test_file:
        Log.i('load test dataset from {}'.format(test_file))
        test_dataset = loader.load(test_file, train=False,
                                   size=16 if utils.is_dev() else None)
    else:
        test_dataset = None

    Log.v('')
    Log.v("initialize ...")
    Log.v('--------------------------------')
    Log.i('# Minibatch-size: {}'.format(batch_size))
    Log.i('# epoch: {}'.format(n_epoch))
    Log.i('# gpu: {}'.format(gpu))
    Log.i('# tagset size: {}'.format(len(loader.tag_map)))
    Log.v('--------------------------------')
    Log.v('')

    # Set up a neural network model
    layers = [
        models.Input(
            word_embeddings=loader.get_embeddings('word'),
            char_embeddings=loader.get_embeddings('char'),
            char_feature_size=50,
            dropout=0.5,
        ),
        models.Recurrent(
            n_layers=2,
            in_size=loader.get_embeddings('word').shape[1] + 50,
            out_size=400,
            dropout=0.5),
        models.Tagger(
            in_size=400 * 2,
            out_size=len(loader.tag_map),
            units=100,
            dropout=0.5) if with_tagging_task else
        models.GoldTagger(out_size=len(loader.tag_map)),
    ]
    if with_parsing_task:
        layers.extend([
            models.Connection(
                tagset_size=len(loader.tag_map),
                tag_embed_size=50,
                dropout=0.5),
            models.Parser(
                in_size=850,
                n_deprels=len(loader.rel_map),
                n_blstm_layers=2,
                lstm_hidden_size=400,
                parser_mlp_units=800,
                dropout=0.50,
                transition_system=transition_system),
        ])
    model = models.MTL(*layers)
    if gpu >= 0:
        framework_utils.set_model_to_device(model, device_id=gpu)

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam(
        alpha=lr, beta1=0.9, beta2=0.999, eps=1e-08)
    optimizer.setup(model)
    if l2_lambda > 0.0:
        optimizer.add_hook(chainer.optimizer.WeightDecay(l2_lambda))
    else:
        l2_lambda = False
    if grad_clip > 0.0:
        optimizer.add_hook(chainer.optimizer.GradientClipping(grad_clip))
    else:
        grad_clip = False
    # optimizer.add_hook(
    #     framework_utils.optimizers.ExponentialDecayAnnealing(
    #         initial_lr=lr, decay_rate=0.75, decay_step=5000, lr_key='alpha'))
    Log.i('optimizer: Adam(alpha={}, beta1=0.9, '
          'beta2=0.999, eps=1e-08), grad_clip={}, '
          'regularization: WeightDecay(lambda={})'
          .format(lr, grad_clip, l2_lambda))

    # Setup a trainer
    trainer = Trainer(optimizer, model,
                      loss_func=model.compute_loss,
                      accuracy_func=model.compute_accuracy)
    trainer.configure(framework_utils.config)
    if test_dataset:
        evaluator = models.Evaluator(loader, test_file, save_to)
        evaluator.add_target(model)
        trainer.attach_callback(evaluator)

    if save_to is not None:
        accessid = Log.getLogger().accessid
        date = Log.getLogger().accesstime.strftime('%Y%m%d')
        trainer.attach_callback(
            framework_utils.callbacks.Saver(
                model,
                basename="{}-{}".format(date, accessid),
                directory=save_to,
                context=dict(App.context,
                             models=[type(layer) for layer in layers],
                             loader=loader)))

    # Start training
    trainer.fit(train_dataset, None,
                batch_size=batch_size,
                epochs=n_epoch,
                validation_data=test_dataset,
                verbose=App.verbose)


if __name__ == "__main__":
    Log.AppLogger.configure(mkdir=True)

    App.add_command('train', train, {
        'batch_size':
        arg('--batchsize', '-b', type=int, default=32,
            help='Number of examples in each mini-batch'),
        'embed_file':
        arg('--embedfile', type=str, default=None,
            help='Pretrained word embedding file'),
        'embed_size':
        arg('--embedsize', type=int, default=100,
            help='Size of embeddings'),
        'gpu':
        arg('--gpu', '-g', type=int, default=-1,
            help='GPU ID (negative value indicates CPU)'),
        'grad_clip':
        arg('--gradclip', type=float, default=5.0,
            help='L2 norm threshold of gradient norm'),
        'l2_lambda':
        arg('--l2', type=float, default=0.0,
            help='Strength of L2 regularization'),
        'lr':
        arg('--lr', type=float, default=0.001,
            help='Learning Rate'),
        'n_epoch':
        arg('--epoch', '-e', type=int, default=20,
            help='Number of sweeps over the dataset to train'),
        'seed':
        arg('--seed', type=int, default=1,
            help='Random seed'),
        'save_to':
        arg('--out', type=str, default=None,
            help='Save model to the specified directory'),
        'tasks':
        arg('--task', type=str, default='tp',
            help='Tasks to train: {t: tagging, p: parsing}'),
        'test_file':
        arg('--validfile', type=str, default=None,
            help='validation data file'),
        'train_file':
        arg('--trainfile', type=str, required=True,
            help='training data file'),
    })

    App.run()
