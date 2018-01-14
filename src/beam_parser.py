import copy
import heapq

import chainer
import chainer.functions as F
import numpy as np
from teras.app import App, arg
import teras.framework.chainer as framework_utils
from teras.framework.chainer.model import MLP
import teras.logging as Log
from teras.training import Trainer

import models
import transition
import utils


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
        self._forward_common(words, chars)
        lengths = [len(x) for x in words]
        gold_actions = args[2]
        return self._forward(lengths, gold_actions)

    def _forward_common(self, words, chars):
        y_input = self.mtl.forward_layer('input', words, chars)
        y_recur = self.mtl.forward_layer('recurrent', y_input)
        y_tagger = self.mtl.forward_layer('tagger', y_recur)
        y_conn = self.mtl.forward_layer('connection', y_recur, y_tagger)
        self.lstm_hs = y_conn

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
        y = self.mtl.forward_layer('parser', features, hs)
        scores = self.mlp(F.concat([
            self.mtl['parser'].cache['h0'],
            self.mtl['parser'].cache['h1'],
            y]))
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
            is_gold_in_beam = any(item[1] for item in beam.items)
            if is_done and is_gold_in_beam:
                """
                case: all states are terminal
                and the gold state does not fall
                """
                gold_index = -1
                for i, item in enumerate(beam.items):
                    if item[1]:
                        gold_index = i
                        break
                beam.items[-1], beam.items[gold_index] = \
                    beam.items[gold_index], beam.items[-1]
            elif not is_gold_in_beam:
                """
                case: the gold state falls out
                """
                if stop_early:
                    beam.items[-1] = gold
                    is_done = True
            if not is_done:
                next_targets.append((beam_index, beam))
        beams[:] = next_targets

    def decode(self, *args):
        return self.parse(args[0], args[1])

    def parse(self, words, chars):
        self._forward_common(words, chars)
        lengths = [len(x) for x in words]
        beams = self._batch_beam_search(lengths, stop_early=False)
        states = [beam.items[0][3] for index, beam in beams]
        heads, labels, _states = zip(*[(state.heads, state.labels, state)
                                       for state in states])
        return heads, labels, _states

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


class ParserEvaluator(models.Evaluator):

    def add_target(self, model):
        self._model = model
        self._has_tagging_task = False
        self._has_parsing_task = True

    def decode(self, xs, ts):
        sentences = xs[3]
        self._buffer['sentences'].extend(sentences)
        heads, labels, states = self._model.decode(*xs)
        self._buffer['heads'].extend(heads)
        self._buffer['labels'].extend(labels)
        return {'heads': heads, 'labels': labels}

    def on_epoch_train_begin(self, data):
        pass

    def on_epoch_train_end(self, data):
        pass


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
    model = BeamParser(model=mtl)
    if gpu >= 0:
        framework_utils.set_model_to_device(model, device_id=gpu)

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam(
        alpha=lr, beta1=0.9, beta2=0.999, eps=1e-08)
    optimizer.setup(model)
    model.mtl.disable_update()
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
    if test_dataset:
        evaluator = ParserEvaluator(loader, test_file, save_to)
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
                             mtl_model=model_file,
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
