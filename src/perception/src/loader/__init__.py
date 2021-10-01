"""Data loaders."""
import json

from loader.cityscapes_loader import CityscapesLoader
from loader.rugd_loader import RUGDLoader


def get_loader(name):
  """get_loader

    :param name:
    """
  return {
      "cityscapes": CityscapesLoader,
      "rugd": RUGDLoader
  }[name]
