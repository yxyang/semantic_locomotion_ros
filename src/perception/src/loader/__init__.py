"""Data loaders."""
import json

from semantic_locomotion.segmentation.loader.cityscapes_loader import cityscapesLoader


def get_loader(name):
  """get_loader

    :param name:
    """
  return {
      "cityscapes": cityscapesLoader,
  }[name]
