"""Data loaders."""
import json

from perception.loader.cityscapes_loader import CityscapesLoader
from perception.loader.rugd_loader import RUGDLoader
from perception.loader.rugd_a1_loader import RUGDA1Loader


def get_loader(name):
  """get_loader

    :param name:
    """
  return {
      "cityscapes": CityscapesLoader,
      "rugd": RUGDLoader,
      "rugd_a1": RUGDA1Loader
  }[name]
