"""Dataloader for RUGD dataset."""
import os

from m1_perception.fchardnet.data_loader import data_loader
from m1_perception.fchardnet.data_loader.data_utils import recursive_glob


class RUGDDataLoader(data_loader.DataLoader):
  """Dataloader for RUGD datset."""
  def __init__(self,
               root_dir,
               batch_size,
               shuffle_between_epoches=True,
               split='train',
               augmentations=None):
    """Initializes the dataloader by finding all image and mask paths."""
    image_files = sorted(
        recursive_glob(os.path.join(root_dir, 'frames', split)))
    mask_files = sorted(
        recursive_glob(os.path.join(root_dir, 'annotations_2d', split)))
    if len(image_files) != len(mask_files):
      raise RuntimeError('Images and annotations do not match.')
    super(RUGDDataLoader,
          self).__init__(batch_size,
                         image_files,
                         mask_files,
                         shuffle_between_epoches=shuffle_between_epoches,
                         augmentations=augmentations)
