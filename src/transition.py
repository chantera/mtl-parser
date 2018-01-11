from array import array
from enum import IntEnum


class State(object):
    __slots__ = ['_num_tokens',
                 '_stack',
                 '_buffer',
                 '_heads',
                 '_labels',
                 '_history']

    def __init__(self, num_tokens):
        self._num_tokens = num_tokens
        self._stack = [0]
        self._buffer = 1
        self._heads = array('h', [-1] * num_tokens)
        self._labels = array('h', [-1] * num_tokens)
        self._history = []

    def advance(self):
        self._buffer += 1

    def push(self, index):
        self._stack.append(index)

    def pop(self):
        return self._stack.pop()

    def add_arc(self, index, head, label):
        self._heads[index] = head
        self._labels[index] = label

    def record(self, action):
        self._history.append(action)

    @property
    def step(self):
        return len(self._history)

    @property
    def num_tokens(self):
        return self._num_tokens

    @property
    def stack_top(self):
        return self._stack[-1]

    def stack(self, position):
        if position < 0:
            return -1
        index = self.stack_size - 1 - position
        return -1 if index < 0 else self._stack[index]

    @property
    def stack_size(self):
        return len(self._stack)

    def statk_empty(self):
        return not(self._stack)

    @property
    def buffer_head(self):
        return self._buffer

    def buffer(self, position):
        if position < 0:
            return -1
        index = self._buffer + position
        return index if index < self._num_tokens else -1

    def buffer_empty(self):
        return self._buffer >= self._num_tokens

    def head(self, index):
        return self._heads[index]

    @property
    def heads(self):
        return self._heads

    def label(self, index):
        return self._labels[index]

    @property
    def labels(self):
        return self._labels

    def leftmost(self, index, check_from=0):
        if (index >= 0 and index < self._num_tokens
                and check_from >= 0 and check_from < index):
            for i in range(check_from, index):
                if self._heads[i] == index:
                    return i
        return -1

    def rightmost(self, index, check_from=-1):
        check_from = self._num_tokens - 1 if check_from == - 1 else check_from
        if (index >= 0 and index < self._num_tokens
                and check_from > index and check_from < self._num_tokens):
            for i in range(check_from, index, -1):
                if self._heads[i] == index:
                    return i
        return -1

    @property
    def history(self):
        return self._history


class GoldState(State):
    __slots__ = State.__slots__ + [
        '_gold_heads',
        '_gold_labels',
    ]

    def __init__(self, gold_heads, gold_labels):
        self._gold_heads = gold_heads
        self._gold_labels = gold_labels
        super().__init__(len(gold_heads))

    def gold_head(self, index):
        return self._gold_heads[index]

    def get_gold_head(self, index, default):
        if index < 0 or index >= self._num_tokens:
            return default
        return self._gold_heads[index]

    def gold_label(self, index):
        return self._gold_labels[index]

    def get_gold_label(self, index, default):
        if index < 0 or index >= self._num_tokens:
            return default
        return self._gold_labels[index]


def projectivize(heads):
    """https://github.com/tensorflow/models/blob/7d30a017fe50b648be6dee544f8059bde52db562/syntaxnet/syntaxnet/document_filters.cc#L296"""  # NOQA
    num_tokens = len(heads)
    while True:
        left = [-1] * num_tokens
        right = [num_tokens] * num_tokens

        for i, head in enumerate(heads):
            l = min(i, head)
            r = max(i, head)
            for j in range(l + 1, r):
                if left[j] < l:
                    left[j] = l
                if right[j] > r:
                    right[j] = r

        deepest_arc = -1
        max_depth = 0
        for i, head in enumerate(heads):
            if head == 0:
                continue
            l = min(i, head)
            r = max(i, head)
            left_bound = max(left[l], left[r])
            right_bound = min(right[l], right[r])

            if l < left_bound or r > right_bound:
                depth = 0
                j = i
                while j != 0:
                    depth += 1
                    j = heads[j]
                if depth > max_depth:
                    deepest_arc = i
                    max_depth = depth

        if deepest_arc == -1:
            return True

        lifted_head = heads[heads[deepest_arc]]
        heads[deepest_arc] = lifted_head


class TransitionSystem(object):

    @classmethod
    def num_action_types(cls):
        pass

    @classmethod
    def num_actions(cls):
        pass


class ArcStandard(TransitionSystem):

    class ActionType(IntEnum):
        SHIFT = 0
        LEFT_ARC = 1
        RIGHT_ARC = 2

    ACTION_NAME = ['SHIFT', 'LEFT_ARC', 'RIGHT_ARC']

    @classmethod
    def shift_action(cls):
        return cls.ActionType.SHIFT

    @classmethod
    def left_arc_action(cls, label):
        return cls.ActionType.LEFT_ARC + (label << 1)

    @classmethod
    def right_arc_action(cls, label):
        return cls.ActionType.RIGHT_ARC + (label << 1)

    @classmethod
    def action_type(cls, action):
        return cls.ActionType(action if action < 1 else 1 + (~action & 1))

    @classmethod
    def action_name(cls, action):
        return cls.ACTION_NAME[
            cls.ActionType(action if action < 1 else 1 + (~action & 1))]

    @classmethod
    def label(cls, action):
        return -1 if action < 1 else (action - 1) >> 1

    @classmethod
    def apply(cls, action, state):
        action_type = cls.action_type(action)
        if action_type == cls.ActionType.SHIFT:
            cls.shift(state)
        elif action_type == cls.ActionType.LEFT_ARC:
            cls.left_arc(state, cls.label(action))
        elif action_type == cls.ActionType.RIGHT_ARC:
            cls.right_arc(state, cls.label(action))
        else:
            raise
        state.record(action)

    """Shift: (s, i|b, A) => (s|i, b, A)"""
    @classmethod
    def shift(cls, state):
        state.push(state.buffer_head)
        state.advance()

    """Left Arc: (s|i|j, b, A) => (s|j, b, A +(j,l,i))"""
    @classmethod
    def left_arc(cls, state, label):
        s0 = state.pop()
        s1 = state.pop()
        state.add_arc(s1, s0, label)
        state.push(s0)

    """Right Arc: (s|i|j, b, A) => (s|i, b, A +(i,l,j))"""
    @classmethod
    def right_arc(cls, state, label):
        s0 = state.pop()
        s1 = state.stack_top
        state.add_arc(s0, s1, label)

    @classmethod
    def is_allowed(cls, action, state):
        action_type = cls.action_type(action)
        if action_type == cls.ActionType.SHIFT:
            return cls.is_allowed_shift(state)
        elif action_type == cls.ActionType.LEFT_ARC:
            return cls.is_allowed_left_arc(state)
        elif action_type == cls.ActionType.RIGHT_ARC:
            return cls.is_allowed_right_arc(state)
        return False

    @classmethod
    def is_allowed_shift(cls, state):
        return not(state.buffer_empty())

    @classmethod
    def is_allowed_left_arc(cls, state):
        return state.stack_size > 2

    @classmethod
    def is_allowed_right_arc(cls, state):
        return state.stack_size > 1

    @classmethod
    def is_terminal(cls, state):
        return state.buffer_empty() and state.stack_size < 2

    @classmethod
    def get_oracle(cls, state):
        if state.stack_size < 2:
            return cls.shift_action()
        s0_id = state.stack(0)
        s1_id = state.stack(1)
        if state.gold_head(s0_id) == s1_id and \
                cls.done_right_children_of(state, s0_id):
            return cls.right_arc_action(state.gold_label(s0_id))
        if state.gold_head(s1_id) == s0_id:
            return cls.left_arc_action(state.gold_label(s1_id))
        return cls.shift_action()

    @staticmethod
    def done_right_children_of(state, head):
        index = state.buffer_head
        while index < state.num_tokens:
            actual_head = state.gold_head(index)
            if actual_head == head:
                return False
            index = head if head > index else index + 1
        return True


class ArcHybrid(ArcStandard):

    """Shift: (s, i|b, A) => (s|i, b, A)"""
    @classmethod
    def shift(cls, state):
        state.push(state.buffer_head)
        state.advance()

    """Left Arc: (s|i, j|b, A) => (s, j|b, A+(j,l,i))"""
    @classmethod
    def left_arc(cls, state, label):
        s0 = state.pop()
        b0 = state.buffer_head
        state.add_arc(s0, b0, label)

    """Right Arc: (s|i|j, b, A) => (s|i, b, A+(i,l,j))"""
    @classmethod
    def right_arc(cls, state, label):
        s0 = state.pop()
        s1 = state.stack_top
        state.add_arc(s0, s1, label)

    @classmethod
    def is_allowed_shift(cls, state):
        return not(state.buffer_empty())

    @classmethod
    def is_allowed_left_arc(cls, state):
        return not(state.buffer_empty()) and state.stack_size > 1

    @classmethod
    def is_allowed_right_arc(cls, state):
        return state.stack_size > 1

    @classmethod
    def get_oracle(cls, state):
        if state.stack_size < 2:
            return cls.shift_action()
        s0_id = state.stack(0)
        if state.gold_head(s0_id) == state.stack(1) and \
                cls.done_right_children_of(state, s0_id):
            return cls.right_arc_action(state.gold_label(s0_id))
        if state.gold_head(s0_id) == state.buffer_head:
            return cls.left_arc_action(state.gold_label(s0_id))
        return cls.shift_action()
