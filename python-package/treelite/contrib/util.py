# coding: utf-8
"""
Miscellaneous utilities
"""

from ..compat import PY3

import shutil

if PY3:
  from tempfile import TemporaryDirectory
else:
  class TemporaryDirectory(object):
    """Context manager for tempfile.mkdtemp()"""
    def __enter__(self):
        self.name = tempfile.mkdtemp()
        return self.name

    def __exit__(self, exc_type, exc_value, traceback):
        shutil.rmtree(self.name)

def _str_decode(str):
  if PY3:
    return str.decode('utf-8')
  else:
    return str

def _str_encode(str):
  if PY3:
    return str.encode('utf-8')
  else:
    return str

__all__ = ['TemporaryDirectory']
