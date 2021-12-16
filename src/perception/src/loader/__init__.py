"""Data loaders."""
import json

from loader.cityscapes_loader import CityscapesLoader
from loader.rugd_loader import RUGDLoader
from loader.rugd_a1_loader import RUGDA1Loader


def get_loader(name):
  """get_loader

    :param name:
    """
  return {
      "cityscapes": CityscapesLoader,
      "rugd": RUGDLoader,
      "rugd_a1": RUGDA1Loader
  }[name]
