"""Generic Dataloader Interface"""
from absl import logging

import numpy as np
from tqdm import tqdm

from m1_perception.fchardnet.data_loader.data_utils import recursive_glob
from m1_perception.fchardnet.model import HardNet


class DataLoader:
  """Generic dataloader for speed model."""
  def __init__(self,
               root_dir,
               batch_size,
               shuffle_between_epoches=True,
               vision_model_ckpt='checkpoints/vision_model/cp-99.ckpt'):
    self._load_files(root_dir)
    logging.info("Loaded {} frames of data.".format(
        self._data['images'].shape[0]))
    self._batch_size = batch_size
    self._shuffle_between_epoches = shuffle_between_epoches
    if self._shuffle_between_epoches:
      self._shuffle_data()
    self._frame_idx = 0

    self._model = HardNet()
    self._model.load_weights(vision_model_ckpt)

  def _load_files(self, root_dir):
    self._data = dict(images=[], cmds=[])
    for filename in tqdm(recursive_glob(root_dir, suffix='npz')):
      curr_file = dict(np.load(filename, allow_pickle=True))
      for key in self._data:
        self._data[key].append(curr_file[key])

    for key in self._data:
      self._data[key] = np.concatenate(self._data[key], axis=0)

  def _shuffle_data(self):
    idx = np.arange(self._data['images'].shape[0])
    np.random.shuffle(idx)
    for key in self._data:
      self._data[key] = self._data[key][idx]

  def next_batch(self):
    """Returns next batch of (embedding, speed) pair."""
    if self._frame_idx + self._batch_size >= self._data['images'].shape[0]:
      self._frame_idx = 0
      if self._shuffle_between_epoches:
        self._shuffle_data()

    images = self._data['images'][self._frame_idx:self._frame_idx +
                                  self._batch_size]
    embeddings = self._model.get_embedding(images)
    speed = self._data['cmds'][
        self._frame_idx:self._frame_idx +  # pylint: disable=E1126
        self._batch_size, 0]
    self._frame_idx += self._batch_size
    return embeddings, speed

  @property
  def num_batches(self):
    return int(self._data['images'].shape[0] / self._batch_size)
