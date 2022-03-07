#!/usr/bin/env python
"""Aligns vision labels and prioperceptive data. """
import os
import datetime

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
  robot_data = dict(
      np.load(open(FLAGS.prioperception_dir, 'rb'), allow_pickle=True))
  vision_data = dict(np.load(open(FLAGS.vision_dir, 'rb'), allow_pickle=True))

  output_data = dict(timestamps=[],
                     mean_speed_commands=[],
                     std_speed_commands=[],
                     mean_actual_speeds=[],
                     std_actual_speeds=[],
                     powers=[],
                     imu_rates=[],
                     imus=[],
                     embeddings=[])

  robot_timestamps = robot_data['timestamp']
  robot_idx = 0

  for vision_idx, vision_timestamp in tqdm(enumerate(
      vision_data['timestamps'])):
    while True:
      curr_diff = abs(robot_timestamps[robot_idx] - vision_timestamp)
      next_diff = abs(robot_timestamps[robot_idx + 1] - vision_timestamp)
      if (curr_diff <= next_diff) or (robot_idx + 2 >= len(robot_timestamps)):
        break
      robot_idx += 1

    if curr_diff < datetime.timedelta(seconds=2):
      for key in output_data:
        if key in vision_data:
          output_data[key].append(vision_data[key][vision_idx])
        else:
          output_data[key].append(robot_data[key][robot_idx])

  for key in output_data:
    output_data[key] = np.array(output_data[key])
  np.savez(os.path.join(FLAGS.output_dir, 'processed_data.npz'), **output_data)


if __name__ == "__main__":
  app.run(main)
