#!/usr/bin/env python
"""Aligns vision labels and prioperceptive data. """
import os

from absl import app
from absl import flags

import numpy as np
from tqdm import tqdm

flags.DEFINE_string('vision_dir', None, 'path to vision data.')
flags.DEFINE_string('prioperception_dir', None, 'path to prioperceptive data.')
flags.DEFINE_string('output_dir', None, 'output dir')

FLAGS = flags.FLAGS


def main(argv):
  del argv  # unused
  robot_data = np.load(open(FLAGS.prioperception_dir, 'rb'), allow_pickle=True)
  vision_data = np.load(open(FLAGS.vision_dir, 'rb'), allow_pickle=True)

  output_data = dict(timestamps=robot_data['timestamp'],
                     foot_force_variance=robot_data['foot_force_difference'],
                     gaits=robot_data['gaits'],
                     speed_commands=robot_data['speed_commands'],
                     actual_speeds=robot_data['actual_speeds'],
                     powers=robot_data['powers'],
                     imu_rates=robot_data['imu_rates'])

  embeddings = []

  vision_timestamps = vision_data['timestamps']
  vision_embeddings = vision_data['embeddings']
  vision_idx = 0

  for robot_timestamp in tqdm(output_data['timestamps']):
    while True:
      curr_diff = abs(vision_timestamps[vision_idx] - robot_timestamp)
      next_diff = abs(vision_timestamps[vision_idx + 1] - robot_timestamp)
      if (curr_diff <= next_diff) or (vision_idx + 2 >=
                                      len(vision_timestamps)):
        break
      vision_idx += 1

    embeddings.append(vision_embeddings[vision_idx])

  output_data["embeddings"] = np.array(embeddings)
  np.savez(os.path.join(FLAGS.output_dir, 'processed_data.npz'), **output_data)


if __name__ == "__main__":
  app.run(main)
