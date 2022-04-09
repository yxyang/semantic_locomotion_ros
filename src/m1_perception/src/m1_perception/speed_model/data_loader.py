"""Generic Dataloader Interface"""
from absl import logging

import numpy as np
from tqdm import tqdm

from m1_perception.fchardnet.data_loader.data_utils import recursive_glob


class DataLoader:
  """Generic dataloader for speed model."""
  def __init__(self, root_dir, batch_size, shuffle_between_epoches=True):
    self._filenames = recursive_glob(root_dir, suffix='npz')
    # Determining total data size
    self._data_size = 0
    for filename in tqdm(self._filenames):
      temp = dict(np.load(filename, allow_pickle=True))
      self._data_size += temp['timestamps'].shape[0]
      del temp
    logging.info("Loaded {} frames of data.".format(self._data_size))
    self._batch_size = batch_size
    self._shuffle_between_epoches = shuffle_between_epoches

    if self._shuffle_between_epoches:
      np.random.shuffle(self._filenames)

    self._file_idx, self._frame_idx = 0, 0
    self._curr_file, self._curr_file_length = None, None
    self._load_file(self._filenames[self._file_idx])

  def _load_file(self, path):
    self._curr_file = dict(np.load(path, allow_pickle=True))
    self._curr_file_length = self._curr_file['timestamps'].shape[0]
    idx = np.arange(self._curr_file_length)
    np.random.shuffle(idx)
    for key in self._curr_file:
      self._curr_file[key] = self._curr_file[key][idx]

  def next_batch(self):
    """Returns next batch of (embedding, speed) pair."""
    if self._frame_idx + self._batch_size >= self._curr_file_length:
      self._file_idx += 1
      if self._file_idx == len(self._filenames):
        if self._shuffle_between_epoches:
          np.random.shuffle(self._filenames)
        self._file_idx = 0

      self._frame_idx = 0
      self._curr_file = dict(
          np.load(self._filenames[self._file_idx], allow_pickle=True))
      self._curr_file_length = self._curr_file['timestamps'].shape[0]

    embeddings = self._curr_file['embeddings'][self.
                                               _frame_idx:self._frame_idx +
                                               self._batch_size]
    speed = self._curr_file['mean_speed_commands'][
        self._frame_idx:self._frame_idx + self._batch_size, 0]
    self._frame_idx += self._batch_size
    return embeddings, speed

  @property
  def num_batches(self):
    return int(self._data_size / self._batch_size)
