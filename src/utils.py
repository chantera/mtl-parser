import os
import re

import teras


_DEVELOP = os.environ.get('ENV', 'production').lower() in ['dev', 'develop']


def is_dev():
    return _DEVELOP


def set_random_seed(seed, gpu=-1):
    import random
    import numpy
    random.seed(seed)
    numpy.random.seed(seed)
    if gpu >= 0:
        try:
            import cupy
            cupy.cuda.runtime.setDevice(gpu)
            cupy.random.seed(seed)
        except Exception as e:
            import teras.logging as Log
            Log.e(str(e))


def load_context(model_file):
    _dir, _file = os.path.split(model_file)
    context_file = os.path.basename(_file).split('.')[0] + '.context'
    context_file = os.path.join(_dir, context_file)
    with open(context_file, 'rb') as f:
        context = teras.base.Context(teras.utils.load(f))
    return context


def to_camelcase(s):
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s)
    s = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s)
    s = re.sub('(_+)_', r'\1', s)
    return s.lower()
