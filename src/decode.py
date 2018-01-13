#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
from teras.app import App, arg
import teras.framework.chainer as framework_utils
import teras.logging as Log

import models
import utils


def decode(
        model_file,
        target_file,
        gpu=-1,
        save_to=None):
    context = utils.load_context(model_file)
    if context.seed is not None:
        utils.set_random_seed(context.seed, gpu)
        Log.i("random seed: {}".format(context.seed))
    framework_utils.set_debug(App.debug)

    loader = context.loader
    Log.i('load test dataset from {}'.format(target_file))
    test_dataset = loader.load(target_file, train=False,
                               size=16 if utils.is_dev() else None)
    Log.i('#samples {}'.format(len(test_dataset)))

    Log.v('')
    Log.v("initialize ...")
    Log.v('--------------------------------')
    Log.i('# gpu: {}'.format(gpu))
    Log.i('# tagset size: {}'.format(len(loader.tag_map)))
    Log.i('# model layers: {}'.format(context.models))
    Log.i('# context: {}'.format(context))
    Log.v('--------------------------------')
    Log.v('')

    models.USE_ORTHONORMAL = False
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
            dropout=0.5) if context.models[2] is models.Tagger else
        models.GoldTagger(out_size=len(loader.tag_map)),
    ]
    if models.Parser in context.models:
        layers.extend([
            models.Connection(
                in_size=400 * 2,
                out_size=800,
                tagset_size=len(loader.tag_map),
                tag_embed_size=50,
                dropout=0.5),
            models.Parser(
                in_size=850,
                n_deprels=len(loader.rel_map),
                n_blstm_layers=1,
                lstm_hidden_size=400,
                parser_mlp_units=800,
                dropout=0.50),
        ])
    model = models.MTL(*layers)
    chainer.serializers.load_npz(model_file, model)
    if gpu >= 0:
        framework_utils.set_model_to_device(model, device_id=gpu)
    # Setup an evaluator
    evaluator = models.Evaluator(loader, target_file, save_to)
    evaluator.add_target(model)

    # Start decoding
    framework_utils.chainer_train_off()
    evaluator.on_epoch_validate_begin({'epoch': 0})
    for batch_index, batch in enumerate(
            test_dataset.batch(context.batch_size, colwise=True,
                               shuffle=False)):
        xs, ts = batch[:-1], batch[-1]
        evaluator.on_batch_begin({'train': False, 'xs': xs, 'ts': ts})
        model(*xs)
        evaluator.on_batch_end({'train': False, 'xs': xs, 'ts': ts})
    evaluator.on_epoch_validate_end({'epoch': 0})


if __name__ == "__main__":
    Log.AppLogger.configure(mkdir=True)

    App.add_command('decode', decode, {
        'gpu':
        arg('--gpu', '-g', type=int, default=-1,
            help='GPU ID (negative value indicates CPU)'),
        'model_file':
        arg('--modelfile', type=str, required=True,
            help='Trained model archive file'),
        'save_to':
        arg('--out', type=str, default=None,
            help='Save results to the specified directory'),
        'target_file':
        arg('--targetfile', type=str, required=True,
            help='Decoding target data file'),
    })

    App.run()
