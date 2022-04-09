"""Generic Dataloader Interface"""
import numpy as np
from PIL import Image
from tqdm import tqdm


def read_image_files(files_list):
  """Reads image files and store in numpy array."""
  results = []
  for image_path in tqdm(files_list):
    image = Image.open(image_path)
    results.append(image.copy())
    image.close()

  # NHWC format
  return np.array(results)


def read_mask_files(files_list):
  """Reads image files and store in numpy array."""
  results = []
  for image_path in tqdm(files_list):
    image = Image.open(image_path)
    results.append(image.copy())
    image.close()

  # NHWC format
  return np.array(results)


def generate_augmented_images(images, masks, augmentations):
  """Augments images and corresponding masks."""
  images_tensor, masks_tensor = [], []
  for image, mask in zip(images, masks):
    if augmentations:
      for augmentation in np.random.choice(augmentations, 2, replace=False):
        image, mask = augmentation(image, mask)

    image = np.array(image, dtype=np.uint8).astype(np.float32) / 255.
    mask = np.array(mask, dtype=np.uint8).astype(np.float32)
    images_tensor.append(image)
    masks_tensor.append(mask)

  return np.stack(images_tensor, axis=0), np.stack(masks_tensor, axis=0)


# @ray.remote
class DataLoader:
  """Generic dataloader for segmentation task."""
  def __init__(self,
               batch_size,
               image_files,
               mask_files,
               shuffle_between_epoches=True,
               augmentations=None):
    self._batch_size = batch_size
    self._images = read_image_files(image_files)
    self._masks = read_mask_files(mask_files)
    self._data_size = len(image_files)
    self._shuffle_between_epoches = shuffle_between_epoches
    self._num_batches = int(self._data_size / self._batch_size)
    self._augmentations = [instance() for instance in augmentations]
    self._start_idx = self._data_size

  def _data_stream(self):
    """Starts datastream for 1 epoch."""
    if self._shuffle_between_epoches:
      idx = np.arange(self._data_size)
      np.random.shuffle(idx)
      self._images = self._images[idx]
      self._masks = self._masks[idx]

    for batch_id in range(self._num_batches):
      start_idx, end_idx = batch_id * self._batch_size, (batch_id +
                                                         1) * self._batch_size
      yield generate_augmented_images(self._images[start_idx:end_idx],
                                      self._masks[start_idx:end_idx],
                                      self._augmentations)

  def next_batch(self):
    """Generates next batch for training."""
    if self._start_idx + self._batch_size > self._data_size:
      # End of current epoch.
      self._start_idx = 0
      if self._shuffle_between_epoches:
        idx = np.arange(self._data_size)
        np.random.shuffle(idx)
        self._images = self._images[idx]
        self._masks = self._masks[idx]

    start_idx = self._start_idx
    end_idx = start_idx + self._batch_size
    self._start_idx += self._batch_size
    result = generate_augmented_images(self._images[start_idx:end_idx],
                                       self._masks[start_idx:end_idx],
                                       self._augmentations)
    return result

  def num_batches(self):
    return self._num_batches
