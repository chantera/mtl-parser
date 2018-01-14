# from collections import defaultdict
# from itertools import accumulate, chain
import copy
import heapq
# import os
# import subprocess
# from tempfile import NamedTemporaryFile

import chainer
import chainer.functions as F
import numpy as np
from teras.app import App, arg
# from teras.base.event import Callback
import teras.framework.chainer as framework_utils
from teras.framework.chainer.model import MLP
import teras.logging as Log
from teras.training import Trainer

import models
import transition
import utils


"""
USE_ORTHONORMAL = True
_orthonormal_initializer = Orthonormal()


class DataLoader(CorpusLoader):

    def __init__(self, rel_map):
        self.rel_map = rel_map

    def map(self, item):
        length = len(item)
        heads, rels = zip(*[
            (token['head'],
             self.rel_map.add(token['deprel'])) for token in item])
        chars = [self._char_transform_one(list(word)) for word in words]
        words = self._word_transform_one(words)
        postags = np.array(postags, dtype=np.int32)
        heads = np.array(heads, dtype=np.int32)
        rels = np.array(rels, dtype=np.int32)

        gold_heads, gold_rels = np.array(heads), np.array(rels)
        transition.projectivize(gold_heads)
        actions, _ = \
            self._extract_gold_transition(gold_heads, gold_rels)
        return
    pass

    def _extract_gold_transition(self, gold_heads, gold_labels):
        features = []
        state = transition.GoldState(gold_heads, gold_labels)
        while not transition.ArcHybrid.is_terminal(state):
            feature = models.Parser.extract_feature(state)
            feature.extend(state.heads)
            features.append(feature)
            action = transition.ArcHybrid.get_oracle(state)
            transition.ArcHybrid.apply(action, state)
        return np.array(state.history, np.int32), np.array(features, np.int32)
"""


class BeamParser(chainer.Chain):
    TransitionSystem = transition.ArcHybrid

    def __init__(self,
                 model,
                 beam_width=8,
                 mlp_units=800,
                 dropout=0.5):
        super().__init__()
        with self.init_scope():
            n_deprels = model['parser'].out_size
            glorotnormal_initializer = chainer.initializers.GlorotNormal()
            self.mtl = model
            self.mlp = MLP([
                MLP.Layer(None, mlp_units, F.relu, dropout,
                          initialW=glorotnormal_initializer),
                MLP.Layer(mlp_units, 1 + 2 * n_deprels,
                          initialW=glorotnormal_initializer),
            ])
        self._beam_width = beam_width
        self.TransitionSystem = model['parser'].TransitionSystem

    def __call__(self, *args):
        words, chars = args[:2]
        lengths = [len(x) for x in words]
        y_input = self.mtl.forward_layer('input', words, chars)
        y_recur = self.mtl.forward_layer('recurrent', y_input)
        y_tagger = self.mtl.forward_layer('tagger', y_recur)
        y_conn = self.mtl.forward_layer('connection', y_recur, y_tagger)
        self.lstm_hs = y_conn
        gold_actions = args[2]
        return self._forward(lengths, gold_actions)

    def _forward(self, sentence_lengths, gold_actions):
        self._gold_actions = gold_actions
        beams = self._batch_beam_search(sentence_lengths, stop_early=True)
        logits = F.vstack([F.concat([F.expand_dims(item[0], axis=0)
                                     for item in reversed(beam.items)], axis=0)
                           for index, beam in beams])
        del beams
        return logits

    def _batch_beam_search(self, sentence_lengths, stop_early=True):
        beams = [(index, Beam(width=self._beam_width,
                              items=[(0.0, 1, 0, transition.State(length))]))
                 for index, length in enumerate(sentence_lengths)]
        targets = beams[:]
        while targets:
            self._advance(targets, stop_early)
        return beams

    def _advance(self, beams, stop_early=True):
        next_targets = []
        features = [self.xp.array([self.mtl['parser'].extract_feature(item[3])
                                   for item in beam.items])
                    for i, beam in beams]
        hs = [self.lstm_hs[i] for i, beam in beams]
        scores = self.mtl.forward_layer('parser', features, hs)
        target_index = -1
        for beam_index, beam in beams:
            is_done = True
            gold = None
            for score, is_gold, step, state in beam.items:
                target_index += 1
                if self.TransitionSystem.is_terminal(state):
                    beam.append((score, is_gold, step, state))
                    continue
                is_done = False
                for action, action_score in enumerate(scores[target_index]):
                    if not self.TransitionSystem.is_allowed(action, state):
                        continue
                    new_score = score + action_score
                    new_is_gold = int(is_gold and action
                                      == self._gold_actions[beam_index][step])
                    new_state = copy.deepcopy(state)
                    self.TransitionSystem.apply(action, new_state)
                    new_item = (new_score, new_is_gold, step + 1, new_state)
                    if new_is_gold:
                        gold = new_item
                    beam.append(new_item)
            beam.prune(key=lambda x: (float(x[0]), x[1], -x[2]))
            if stop_early and all(gold[3] is not item[3]
                                  for item in beam.items):
                beam.items[-1] = gold
                is_done = True
            if not is_done:
                next_targets.append((beam_index, beam))
            # @TODO: check all done and gold does not fall
        beams[:] = next_targets

    def compute_loss(self, ys, ts):
        loss = F.softmax_cross_entropy(
            ys,
            self.xp.array(np.hstack(ts)),
            normalize=True,
            reduce='mean')
        return loss

    def compute_accuracy(self, ys, ts):
        accuracy = F.accuracy(
            ys,
            self.xp.array(np.hstack(ts)))
        return accuracy


class Beam(object):

    def __init__(self, width, items=None):
        if items:
            assert len(items) <= width
        else:
            items = []
        self._width = width
        self._items = items
        self._new_items = []

    def append(self, item):
        self._new_items.append(item)

    def prune(self, key=None):
        self._items[:] = heapq.nlargest(self._width, self._new_items, key)
        self._new_items = []

    @property
    def width(self):
        return self._width

    @property
    def items(self):
        return self._items


def load_mtl(loader, context, model_file):
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
    return model


def train(
        model_file,
        train_file,
        test_file=None,
        n_epoch=20,
        batch_size=32,
        lr=0.001,
        l2_lambda=0.0001,
        gpu=-1,
        save_to=None):
    context = utils.load_context(model_file)
    if context.seed is not None:
        utils.set_random_seed(context.seed, gpu)
        Log.i("random seed: {}".format(context.seed))
    framework_utils.set_debug(App.debug)

    loader = context.loader
    loader.set_mode('beamparsing')
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
    Log.v('--------------------------------')
    Log.v('')

    # Set up a neural network model
    mtl = load_mtl(loader, context, model_file)
    mtl.disable_update()
    model = BeamParser(model=mtl)
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
    Log.i('optimizer: Adam(alpha={}, beta1=0.9, beta2=0.999, eps=1e-08), '
          'regularization: WeightDecay(lambda={})'.format(lr, l2_lambda))

    # Setup a trainer
    trainer = Trainer(optimizer, model,
                      loss_func=model.compute_loss,
                      accuracy_func=model.compute_accuracy)
    trainer.configure(framework_utils.config)
    # if test_dataset:
    #     evaluator = models.Evaluator(loader, test_file, save_to)
    #     evaluator.add_target(model)
    #     trainer.attach_callback(evaluator)

    """
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
    """

    # Start training
    trainer.fit(train_dataset, None,
                batch_size=batch_size,
                epochs=n_epoch,
                validation_data=test_dataset,
                verbose=App.verbose)


# class Evaluator(Callback):
#     PERL = '/usr/bin/perl'
#     SCRIPT = os.path.join(
#         os.path.abspath(os.path.dirname(__file__)), 'eval.pl')
# 
#     def __init__(self, loader, gold_file, out_dir=None,
#                  name='evaluator', **kwargs):
#         super(Evaluator, self).__init__(name, **kwargs)
#         self._loader = loader
#         self._gold_file = os.path.abspath(os.path.expanduser(gold_file))
#         self._out_dir = os.path.abspath(os.path.expanduser(out_dir)) \
#             if out_dir is not None else None
#         basename = os.path.basename(gold_file)
#         accessid = Log.getLogger().accessid
#         date = Log.getLogger().accesstime.strftime('%Y%m%d')
#         self._out_file_format = date + "-" + accessid + ".{}." + basename
# 
#     def add_target(self, model):
#         self._model = model
#         self._has_tagging_task = 'tagger' in model \
#             and not isinstance(model._layers['tagger'], GoldTagger)
#         self._has_parsing_task = 'parser' in model
# 
#     def decode(self, xs, ts):
#         sentences = xs[4]
#         self._buffer['sentences'].extend(sentences)
#         results = self._model.decode(*xs)
# 
#         if self._has_tagging_task:
#             tags = results['tags']
#             tags.to_cpu()
#             self._buffer['postags'].extend(tags.data)
#         #     true_tags = ts.T[0]
#         #     for i, (p_tags, t_tags) in enumerate(
#         #             zip(tags_batch, true_tags)):
#         #         # @TODO: evaluate tags
# 
#         if self._has_parsing_task:
#             self._buffer['heads'].extend(results['heads'])
#             self._buffer['labels'].extend(results['deprels'])
# 
#         return results
# 
#     def flush(self, out):
#         self._loader.write_conll(
#             out,
#             self._buffer['sentences'],
#             self._buffer['heads'],
#             self._buffer['labels'],
#             self._buffer['postags']
#             if len(self._buffer['postags']) > 0 else None)
# 
#     def report(self, target):
#         command = [self.PERL, self.SCRIPT,
#                    '-g', self._gold_file, '-s', target, '-q']
#         Log.v("exec command: {}".format(' '.join(command)))
#         p = subprocess.run(command,
#                            stdout=subprocess.PIPE,
#                            stderr=subprocess.PIPE,
#                            encoding='utf-8')
#         if p.returncode == 0:
#             Log.i("[evaluation]\n" + p.stdout.rstrip())
#         else:
#             Log.i("[evaluation] ERROR: " + p.stderr.rstrip())
# 
#     def on_batch_begin(self, data):
#         pass
# 
#     def on_batch_end(self, data):
#         if data['train']:
#             return
#         self.decode(data['xs'], data['ts'])
# 
#     def on_epoch_train_end(self, data):
#         records = self._model.get_records()
#         if self._has_tagging_task:
#             Log.i("tag_loss: {:.8f}, tag_accuracy: {:.8f}".format(
#                 records['tag_loss'], records['tag_accuracy']))
#         if self._has_parsing_task:
#             Log.i("action_loss: {:.8f}, action_accuracy: {:.8f}".format(
#                 records['action_loss'], records['action_accuracy']))
#         self._model.reset_records()
# 
#     def on_epoch_validate_begin(self, data):
#         self._buffer = {
#             'sentences': [],
#             'postags': [],
#             'heads': [],
#             'labels': [],
#         }
# 
#     def on_epoch_validate_end(self, data):
#         self.on_epoch_train_end(data)
#         if not self._has_parsing_task:
#             return
#         if self._out_dir is not None:
#             file = os.path.join(
#                 self._out_dir, self._out_file_format.format(data['epoch']))
#             self.flush(file)
#             self.report(file)
#         else:
#             f = NamedTemporaryFile(mode='w')
#             try:
#                 """
#                 Note:
#                 Whether the name can be used to open the file a second time,
#                 while the named temporary file is still open, varies across
#                 platforms (it can be so used on Unix; it cannot on Windows
#                 NT or later)
#                 """
#                 self.flush(f.name)
#                 self.report(f.name)
#             finally:
#                 f.close()


if __name__ == "__main__":
    Log.AppLogger.configure(mkdir=True)

    App.add_command('train', train, {
        'batch_size':
        arg('--batchsize', '-b', type=int, default=32,
            help='Number of examples in each mini-batch'),
        'gpu':
        arg('--gpu', '-g', type=int, default=-1,
            help='GPU ID (negative value indicates CPU)'),
        'l2_lambda':
        arg('--l2', type=float, default=0.0001,
            help='Strength of L2 regularization'),
        'lr':
        arg('--lr', type=float, default=0.001,
            help='Learning Rate'),
        'model_file':
        arg('--modelfile', type=str, required=True,
            help='Trained model archive file'),
        'n_epoch':
        arg('--epoch', '-e', type=int, default=20,
            help='Number of sweeps over the dataset to train'),
        'save_to':
        arg('--out', type=str, default=None,
            help='Save model to the specified directory'),
        'test_file':
        arg('--validfile', type=str, default=None,
            help='validation data file'),
        'train_file':
        arg('--trainfile', type=str, required=True,
            help='training data file'),
    })

    App.run()
