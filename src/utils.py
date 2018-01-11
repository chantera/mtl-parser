import os
import re


_DEVELOP = os.environ.get('ENV', 'production').lower() in ['dev', 'develop']


def is_dev():
    return _DEVELOP


def to_camelcase(s):
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s)
    s = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s)
    s = re.sub('(_+)_', r'\1', s)
    return s.lower()
