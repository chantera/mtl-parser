from array import array
import weakref

# from chainer import cuda
import numpy as np
from teras.dataset.loader import CorpusLoader
from teras.io.reader import ConllReader
from teras.preprocessing import text

import transition
import models


_CHAR_PADDING = "<PAD>"


class DataLoader(CorpusLoader):
    PAD_CHAR_INDEX = 1

    def __init__(self,
                 word_embed_size=100,
                 char_embed_size=10,
                 word_embed_file=None,
                 word_preprocess=lambda x: x.lower(),
                 word_unknown="<UNK>",
                 embed_dtype='float32',
                 convert_cc_head=False):
        super(DataLoader, self).__init__(reader=ConllReader())
        self.use_pretrained = word_embed_file is not None
        self.add_processor(
            'word',
            embed_file=word_embed_file,
            embed_size=word_embed_size,
            embed_dtype=embed_dtype,
            preprocess=word_preprocess, unknown=word_unknown,
            min_frequency=1)
        self.add_processor(
            'char', embed_size=char_embed_size,
            embed_dtype=embed_dtype,
            preprocess=False)
        self.tag_map = text.Vocab()
        self.rel_map = text.Vocab()
        DataLoader.PAD_CHAR_INDEX = self.get_processor('char') \
            .fit_transform_one([_CHAR_PADDING])[-1]

    def map(self, item):
        """
        item -> (words, chars, features, weakref_of_gold_postags,
                 tokens,  # without projectivization for evaluation
                 (postags, actions))
        """
        words, postags, heads, rels = zip(*[
            (token['form'],
            self.tag_map.add(token['postag']),
            token['head'],
            self.rel_map.add(token['deprel'])) for token in item])
        chars = [self._char_transform_one(list(word)) for word in words]
        words = self._word_transform_one(words)
        postags = np.array(postags, dtype=np.int32)
        heads = np.array(heads, dtype=np.int32)
        rels = np.array(rels, dtype=np.int32)

        gold_heads, gold_rels = np.array(heads), np.array(rels)
        transition.projectivize(gold_heads)
        actions, features = self._extract_gold_transition(gold_heads, gold_rels)

        sample = (words, chars, features, weakref.ref(postags),
                  item if not self._train else None,  # for eval
                  (postags, actions))
        return sample

    def _extract_gold_transition(self, gold_heads, gold_labels):
        # actions = []
        features = []
        state = transition.GoldState(gold_heads, gold_labels)
        while not transition.ArcHybrid.is_terminal(state):
            feature = models.Parser.extract_feature(state)
            feature.extend(state.heads)
            features.append(feature)
            action = transition.ArcHybrid.get_oracle(state)
            # actions.append(int(action))
            transition.ArcHybrid.apply(action, state)
        return np.array(state.history, np.int32), np.array(features, np.int32)

    def load(self, file, train=False, size=None):
        self.set_train(train)
        return super(DataLoader, self).load(file, train, size)

    def set_train(self, train=True):
        self._train = train
        if train:
            if not self.use_pretrained:
                self._word_transform_one = \
                    self.get_processor('word').fit_transform_one
            else:
                self._word_transform_one = \
                    self.get_processor('word').transform_one
            self._char_transform_one = \
                self.get_processor('char').fit_transform_one
        else:
            self._word_transform_one = \
                self.get_processor('word').transform_one
            self._char_transform_one = \
                self.get_processor('char').transform_one

    def write_conll(self, file, sentences, heads, labels, postags=None):
        with open(file, 'w') as f:
            for i, tokens in enumerate(sentences):
                _iter = enumerate(tokens)
                next(_iter)
                for j, token in _iter:
                    line = '\t'.join([
                        str(token['id']),
                        token['form'],
                        token['lemma'],
                        token['cpostag'],
                        self.tag_map.lookup(postags[i][j])
                        if postags is not None else token['postag'],
                        token['feats'],
                        str(heads[i][j]),
                        self.rel_map.lookup(labels[i][j]),
                        token['phead'],
                        token['pdeprel'],
                    ])
                    f.write(line + '\n')
                f.write('\n')
