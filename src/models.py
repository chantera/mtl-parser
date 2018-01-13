from collections import defaultdict
from itertools import accumulate, chain
import os
import subprocess
from tempfile import NamedTemporaryFile

import chainer
import chainer.functions as F
import numpy as np
from teras.base.event import Callback
from teras.framework.chainer.initializers import Orthonormal
from teras.framework.chainer.model import BiLSTM, CharCNN, MLP
import teras.logging as Log

import transition
import utils


USE_ORTHONORMAL = True
_orthonormal_initializer = Orthonormal()
_glorotnormal_initializer = chainer.initializers.GlorotNormal()


def _get_rnn_initializer():
    return _orthonormal_initializer \
        if USE_ORTHONORMAL else _glorotnormal_initializer


if utils.is_dev():
    """
    use _orthonormal_initializer instead of _glorotnormal_initializer
    because _orthonormal_initializer takes time
    """
    USE_ORTHONORMAL = False


def _flatten(list_of_lists):
    return chain.from_iterable(list_of_lists)


class TaskWorker(object):

    def compute_loss(self, ys, ts):
        xp = chainer.cuda.get_array_module(ys, ts)
        return chainer.Variable(xp.array(0))

    def compute_accuracy(self, ys, ts):
        xp = chainer.cuda.get_array_module(ys, ts)
        return chainer.Variable(xp.array(0))


class _InputLayer(chainer.link.Chain):
    pass


class _RecurrentLayer(chainer.link.Chain):
    pass


class _Tagger(chainer.link.Chain, TaskWorker):
    pass


class _ConnectionLayer(chainer.link.Chain):
    pass


class _Parser(chainer.link.Chain, TaskWorker):
    pass


class MTL(chainer.link.ChainList, TaskWorker):
    _LAYER_CLASSES = [_InputLayer,
                      _RecurrentLayer,
                      _Tagger,
                      _ConnectionLayer,
                      _Parser]

    def __init__(self, *layers):
        self._layers = defaultdict(lambda: None)
        classes = {cls: utils.to_camelcase(
            cls.__name__.lstrip('_')).split('_')[0]
            for cls in self._LAYER_CLASSES}
        for layer in layers:
            for cls, name in classes.items():
                if isinstance(layer, cls):
                    self._layers[name] = layer
                    break
            else:
                name = utils.to_camelcase(layer.__class__.__name__)
                self._layers[name] = layer
        super().__init__(*layers)
        self.reset_records()

    def __contains__(self, layer):
        return layer in self._layers

    def __call__(self, *args):
        words, chars, features = args[:3]
        y = []
        y_cache = {'tagger': None, 'conn': None, 'parser': None}
        y_input = self._layers['input'](words, chars)
        y_recur = self._layers['recurrent'](y_input)
        if isinstance(self._layers['tagger'], GoldTagger):
            y_tagger = self._layers['tagger'](y_recur, args[3])
            y.append(y_tagger)
        else:
            y_tagger = self._layers['tagger'](y_recur)
            y.append(y_tagger)
            y_cache['tagger'] = y_tagger
        if self._layers['parser'] is not None:
            assert self._layers['connection'] is not None
            y_conn = self._layers['connection'](y_recur, y_tagger)
            y_cache['conn'] = y_conn
            y_parser = self._layers['parser'](features, y_conn)
            y.append(y_parser)
            y_cache['parser'] = y_parser
        self._y_cache = y_cache
        return y if len(y) > 1 else y[0]

    def compute_loss(self, ys, ts):
        ts_tags, ts_actions = ts.T
        loss = []
        if self._y_cache['tagger'] is not None:
            tag_loss = self._layers['tagger'] \
                .compute_loss(self._y_cache['tagger'], ts_tags)
            self._record('tag_loss', tag_loss)
            loss.append(tag_loss)
        if self._y_cache['parser'] is not None:
            action_loss = self._layers['parser'] \
                .compute_loss(self._y_cache['parser'], ts_actions)
            self._record('action_loss', action_loss)
            loss.append(action_loss)
        return F.sum(F.hstack(loss))

    def compute_accuracy(self, ys, ts):
        ts_tags, ts_actions = ts.T
        accuracy = []
        if self._y_cache['tagger'] is not None:
            tag_accuracy = self._layers['tagger'] \
                .compute_accuracy(self._y_cache['tagger'], ts_tags)
            self._record('tag_accuracy', tag_accuracy)
            accuracy.append(tag_accuracy)
        if self._y_cache['parser'] is not None:
            action_accuracy = self._layers['parser'] \
                .compute_accuracy(self._y_cache['parser'], ts_actions)
            self._record('action_accuracy', action_accuracy)
            accuracy.append(action_accuracy)
        return F.mean(F.hstack(accuracy))

    def reset_records(self):
        self._records = {
            'tag_loss': [],
            'tag_accuracy': [],
            'action_loss': [],
            'action_accuracy': [],
        }

    def _record(self, name, value):
        self._records[name].append(float(value))

    def get_records(self):
        return {key: np.mean(value) for key, value in self._records.items()}

    def decode(self, *args):
        results = {}
        words, chars, features = args[:3]
        results['tags'] = self._layers['tagger'].decode()
        if self._layers['parser'] is not None:
            heads, deprels, states = \
                self._layers['parser'].parse(self._y_cache['conn'])
            results['heads'] = heads
            results['deprels'] = deprels
            results['actions'] = [state.history for state in states]
        return results


class Input(_InputLayer):

    def __init__(self,
                 word_embeddings,
                 char_embeddings,
                 char_feature_size=50,
                 dropout=0.0):
        super().__init__()
        self._dropout_ratio = dropout
        with self.init_scope():
            w_embed_shape = word_embeddings.shape
            self.word_embed = chainer.links.EmbedID(
                in_size=w_embed_shape[0],
                out_size=w_embed_shape[1],
                initialW=word_embeddings
            )
            self.char_cnn = CharCNN(char_embeddings,
                                    out_size=char_feature_size,
                                    pad_id=1, window_size=5)

    def __call__(self, words, chars):
        boundaries = list(accumulate(len(x) for x in words[:-1]))
        xp = chainer.cuda.get_array_module(words[0])
        words_batch = xp.concatenate(words, axis=0)
        chars_batch = list(_flatten(chars))
        xs_words = self.word_embed(self.xp.array(words_batch))
        xs_chars = self.char_cnn(chars_batch)
        xs = F.concat((
            F.dropout(xs_words, self._dropout_ratio),
            F.dropout(xs_chars, self._dropout_ratio)))
        xs = F.split_axis(xs, boundaries, axis=0)
        return xs


class Recurrent(_RecurrentLayer):

    def __init__(self,
                 n_layers,
                 in_size,
                 out_size,
                 dropout=0.5):
        super().__init__()
        with self.init_scope():
            self.bilstm = BiLSTM(n_layers, in_size, out_size, dropout,
                                 initialW=_get_rnn_initializer())

    def __call__(self, xs):
        return self.bilstm(xs)


class Tagger(_Tagger):

    def __init__(self,
                 in_size,
                 out_size,
                 units,
                 dropout=0.5):
        super().__init__()
        with self.init_scope():
            if not hasattr(units, '__iter__'):
                units = [units]
            layers = [MLP.Layer(
                None, unit, F.relu, dropout,
                initialW=_glorotnormal_initializer) for unit in units]
            layers.append(MLP.Layer(
                units[-1], out_size, initialW=_glorotnormal_initializer))
            self.mlp = MLP(layers)

    def __call__(self, xs):
        if not isinstance(xs, chainer.Variable):
            xs = F.pad_sequence(xs)
        ys = self.mlp(xs)
        self._ys = ys
        return ys

    def compute_loss(self, ys, ts):
        batch_size, max_length, n_classes = ys.shape
        ts = F.pad_sequence(ts, padding=-1)
        if not self._cpu:
            ts.to_gpu()
        loss = F.softmax_cross_entropy(
            F.reshape(ys, (batch_size * max_length, n_classes)),
            F.reshape(ts, (batch_size * max_length,)),
            normalize=True,
            reduce='mean',
            ignore_label=-1)
        return loss

    def compute_accuracy(self, ys, ts):
        batch_size, max_length, n_classes = ys.shape
        ts = F.pad_sequence(ts, padding=-1)
        if not self._cpu:
            ts.to_gpu()
        accuracy = F.accuracy(
            F.reshape(ys, (batch_size * max_length, n_classes)),
            F.reshape(ts, (batch_size * max_length,)),
            ignore_label=-1)
        return accuracy

    def decode(self):
        tags = F.argmax(self._ys, axis=2)
        return tags


class GoldTagger(_Tagger):

    def __init__(self, out_size):
        super().__init__()
        self.identity = self.xp.identity(out_size)

    def __call__(self, xs, ts):
        if not isinstance(xs, chainer.Variable):
            xs = F.pad_sequence(xs)
        ys = (self.xp.array(self.identity[t_ref()]) for t_ref in ts)
        ys = F.pad_sequence(ys)
        self._ys = ys
        return ys

    def decode(self):
        tags = F.argmax(self._ys, axis=2)
        return tags


class Connection(_ConnectionLayer):

    def __init__(self,
                 in_size,
                 out_size,
                 tagset_size,
                 tag_embed_size=50,
                 dropout=0.5):
        super().__init__()
        self._dropout_ratio = dropout
        with self.init_scope():
            self.tag_embed = chainer.links.EmbedID(
                in_size=tagset_size,
                out_size=tag_embed_size,
                initialW=None
            )
            self.bilstm = BiLSTM(1, in_size, out_size // 2, dropout,
                                 initialW=_get_rnn_initializer())

    def __call__(self, hs, tag_scores):
        H = []
        tags = F.argmax(tag_scores, axis=2)
        batchsize = len(hs)
        for i in range(batchsize):
            xs_tags = self.tag_embed(tags[i, :hs[i].shape[0]])
            H.append(F.concat([
                F.dropout(hs[i], self._dropout_ratio),
                F.dropout(xs_tags, self._dropout_ratio)]))
        return H


class Parser(_Parser):
    TransitionSystem = transition.ArcHybrid

    def __init__(self,
                 in_size,
                 n_deprels=43,
                 n_blstm_layers=1,
                 lstm_hidden_size=400,
                 parser_mlp_units=800,
                 dropout=0.5):
        super().__init__()
        with self.init_scope():
            self.parser_blstm = BiLSTM(
                n_layers=n_blstm_layers,
                in_size=in_size,
                out_size=lstm_hidden_size,
                dropout=dropout,
                initialW=_get_rnn_initializer()
            )
            self.pad_embed = chainer.links.EmbedID(
                in_size=4,
                out_size=in_size,
            )
            self.pad_linear = chainer.links.Linear(
                in_size=in_size,
                out_size=lstm_hidden_size * 2,
            )
            self.parser_mlp = MLP([
                MLP.Layer(None, parser_mlp_units, F.relu, dropout,
                          initialW=_glorotnormal_initializer),
                MLP.Layer(parser_mlp_units, 1 + 2 * n_deprels,
                          initialW=_glorotnormal_initializer),
            ])
            self.span_embed = chainer.links.EmbedID(
                in_size=4,
                out_size=10,
                initialW=_glorotnormal_initializer,
                ignore_label=-1,
            )
            self._lstm_hidden_size = lstm_hidden_size

    def __call__(self, features, hs):
        self.hs = self.parser_blstm(hs)
        self.pads = F.tanh(self.pad_linear(self.pad_embed(self.xp.arange(4))))
        fs = F.vstack(self._populate_features(features[i], i)
                      for i in range(len(features)))
        fs = F.stack(F.split_axis(fs, fs.shape[0] // 4, axis=0), axis=0)
        fs = F.reshape(fs, (fs.shape[0], -1))
        action_scores = self.parser_mlp(F.reshape(fs, (fs.shape[0], -1)))
        return action_scores

    def _populate_features(self, features, batch_index):
        _feats = self.xp.array(features[:, :4].flatten())
        mask = _feats == -1
        fs = F.embed_id(_feats, self.hs[batch_index], ignore_label=-1)
        fs += F.tile(self.pads, (len(features), 1)) \
            * self.xp.expand_dims(mask, axis=1)
        return fs

    def compute_loss(self, ys, ts):
        true_actions = np.hstack(ts)
        if not self._cpu:
            true_actions = self.xp.array(true_actions)
        loss = F.softmax_cross_entropy(
            ys, true_actions, normalize=True, reduce='mean')
        return loss

    def compute_accuracy(self, ys, ts):
        true_actions = np.hstack(ts)
        if not self._cpu:
            true_actions = self.xp.array(true_actions)
        accuracy = F.accuracy(ys, true_actions)
        return accuracy

    @staticmethod
    def extract_feature(state):
        feature = [state.stack(2),
                   state.stack(1),
                   state.stack(0),
                   state.buffer(0)]
        return feature

    def parse(self, hs):
        self.hs = self.parser_blstm(hs)
        self.pads = F.tanh(self.pad_linear(self.pad_embed(self.xp.arange(4))))
        states = [(sent_idx, transition.State(h.shape[0]))
                  for sent_idx, h in enumerate(hs)]
        _states = states[:]
        while len(_states) > 0:
            features = []
            fs = []
            for sent_idx, state in _states:
                feature = self.xp.array([self.extract_feature(state)])
                features.append(feature)
                fs.append(self._populate_features(feature, sent_idx))
            fs = F.vstack(fs)
            fs = F.stack(F.split_axis(fs, fs.shape[0] // 4, axis=0), axis=0)
            fs = F.reshape(fs, (fs.shape[0], -1))

            action_scores = self.parser_mlp(fs)
            action_scores.to_cpu()
            action_scores = action_scores.data
            for i in range(len(_states) - 1, -1, -1):
                _, state = _states[i]
                best_action, best_score = -1, -np.inf
                for action, score in enumerate(action_scores[i]):
                    if score > best_score and \
                            self.TransitionSystem.is_allowed(action, state):
                        best_action, best_score = action, score
                self.TransitionSystem.apply(best_action, state)
                if self.TransitionSystem.is_terminal(state):
                    del _states[i]

        heads, labels, _states = zip(*[(state.heads, state.labels, state)
                                       for i, state in states])
        return heads, labels, _states


class Evaluator(Callback):
    PERL = '/usr/bin/perl'
    SCRIPT = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'eval.pl')

    def __init__(self, loader, gold_file, out_dir=None,
                 name='evaluator', **kwargs):
        super(Evaluator, self).__init__(name, **kwargs)
        self._loader = loader
        self._gold_file = os.path.abspath(os.path.expanduser(gold_file))
        self._out_dir = os.path.abspath(os.path.expanduser(out_dir)) \
            if out_dir is not None else None
        basename = os.path.basename(gold_file)
        accessid = Log.getLogger().accessid
        date = Log.getLogger().accesstime.strftime('%Y%m%d')
        self._out_file_format = date + "-" + accessid + ".{}." + basename

    def add_target(self, model):
        self._model = model
        self._has_tagging_task = 'tagger' in model \
            and not isinstance(model._layers['tagger'], GoldTagger)
        self._has_parsing_task = 'parser' in model

    def decode(self, xs, ts):
        sentences = xs[4]
        self._buffer['sentences'].extend(sentences)
        results = self._model.decode(*xs)

        if self._has_tagging_task:
            tags = results['tags']
            tags.to_cpu()
            self._buffer['postags'].extend(tags.data)
        #     true_tags = ts.T[0]
        #     for i, (p_tags, t_tags) in enumerate(
        #             zip(tags_batch, true_tags)):
        #         # @TODO: evaluate tags

        if self._has_parsing_task:
            self._buffer['heads'].extend(results['heads'])
            self._buffer['labels'].extend(results['deprels'])

        return results

    def flush(self, out):
        self._loader.write_conll(
            out,
            self._buffer['sentences'],
            self._buffer['heads'],
            self._buffer['labels'],
            self._buffer['postags']
            if len(self._buffer['postags']) > 0 else None)

    def report(self, target):
        command = [self.PERL, self.SCRIPT,
                   '-g', self._gold_file, '-s', target, '-q']
        Log.v("exec command: {}".format(' '.join(command)))
        p = subprocess.run(command,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE,
                           encoding='utf-8')
        if p.returncode == 0:
            Log.i("[evaluation]\n" + p.stdout.rstrip())
        else:
            Log.i("[evaluation] ERROR: " + p.stderr.rstrip())

    def on_batch_end(self, data):
        if data['train']:
            return
        self.decode(data['xs'], data['ts'])

    def on_epoch_train_end(self, data):
        records = self._model.get_records()
        if self._has_tagging_task:
            Log.i("tag_loss: {:.8f}, tag_accuracy: {:.8f}".format(
                records['tag_loss'], records['tag_accuracy']))
        if self._has_parsing_task:
            Log.i("action_loss: {:.8f}, action_accuracy: {:.8f}".format(
                records['action_loss'], records['action_accuracy']))
        self._model.reset_records()

    def on_epoch_validate_begin(self, data):
        self._buffer = {
            'sentences': [],
            'postags': [],
            'heads': [],
            'labels': [],
        }

    def on_epoch_validate_end(self, data):
        self.on_epoch_train_end(data)
        if not self._has_parsing_task:
            return
        if self._out_dir is not None:
            file = os.path.join(
                self._out_dir, self._out_file_format.format(data['epoch']))
            self.flush(file)
            self.report(file)
        else:
            f = NamedTemporaryFile(mode='w')
            try:
                """
                Note:
                Whether the name can be used to open the file a second time,
                while the named temporary file is still open, varies across
                platforms (it can be so used on Unix; it cannot on Windows
                NT or later)
                """
                self.flush(f.name)
                self.report(f.name)
            finally:
                f.close()
