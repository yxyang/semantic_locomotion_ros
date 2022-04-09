"""Utilities for data processing."""

import os

def recursive_glob(rootdir=".", suffix=""):
  """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
  return [
      os.path.join(looproot, filename)
      for looproot, _, filenames in os.walk(rootdir) for filename in filenames
      if filename.endswith(suffix)
  ]
