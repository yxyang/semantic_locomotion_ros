#!/usr/bin/env python
"""Generates training data from proprioceptive and perception data."""
import datetime
import multiprocessing
import os
import pickle

from absl import app
from absl import flags

import numpy as np
import rospy
from tqdm import tqdm

import pybullet

flags.DEFINE_string('logdir', None, 'directory for proprioceptive data.')
flags.DEFINE_string('output_dir', None,
                    'where to dump processed training data.')
flags.DEFINE_bool(
    'filter_out_inconsistent_gaits', True,
    'whether to filter out data where the next 10 steps have inconsistent'
    ' gaits.')
FLAGS = flags.FLAGS


def analyze_individual_file(args):
  """Load data from individual trajectory logs."""
  base_dir, filename = args
  end_ts = datetime.datetime.strptime(filename[4:-4], '%Y_%m_%d_%H_%M_%S')
  speed_commands = []
  steer_commands = []
  timestamps = []
  foot_forces = []
  foot_contacts = []
  foot_phases = []
  actual_speeds = []
  powers = []
  imu_rates = []
  imus = []
  image_embeddings = []
  try:
    robot_states = pickle.load(open(os.path.join(base_dir, filename), 'rb'))
    for frame in robot_states[:-1]:
      foot_phases.append(frame['gait_generator_phase'])
      speed_commands.append(frame['desired_speed'][0])
      steer_commands.append(frame['desired_speed'][1])
      foot_forces.append(frame['foot_forces'])
      foot_contacts.append(frame['foot_contacts'])
      actual_speeds.append(frame['base_vels_body_frame'])
      image_embeddings.append(frame['image_embedding'])
      imus.append(
          pybullet.getEulerFromQuaternion(frame['base_quat_ground_frame']))
      power = np.maximum(
          frame['motor_torques'] * frame['motor_vels'] +
          0.3 * frame['motor_torques']**2, 0)
      powers.append(power)
      imu_rates.append(frame['base_rpy_rate'])

      delta_time_s = datetime.timedelta(seconds=frame['timestamp'] -
                                        robot_states[-1]['timestamp'])
      timestamps.append(end_ts + delta_time_s)
  except pickle.UnpicklingError:
    rospy.loginfo("Corrupted file: {}".format(filename))
    return None

  return dict(foot_phases=foot_phases,
              speed_commands=speed_commands,
              steer_commands=steer_commands,
              timestamps=timestamps,
              foot_contacts=foot_contacts,
              foot_forces=foot_forces,
              actual_speeds=actual_speeds,
              powers=powers,
              imu_rates=imu_rates,
              imus=imus,
              image_embeddings=image_embeddings)


def load_proprioceptive_data(logdir):
  """Loads proprioceptive data.

  Returns a dict of arrays.
  """
  filenames = sorted(os.listdir(logdir))
  args = []
  for filename in filenames:
    args.append((logdir, filename))

  with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as p:
    results = list(
        tqdm(p.imap(analyze_individual_file, args), total=len(filenames)))

  all_results = {}
  for key in results[0]:
    all_results[key] = np.concatenate(
        [result[key] for result in results if result is not None], axis=0)

  # Fix timestamp misalignment
  count = 0
  for idx in tqdm(range(1, len(all_results['timestamps']))):
    if all_results['timestamps'][idx] < all_results['timestamps'][idx - 1]:
      all_results['timestamps'][idx] = all_results['timestamps'][
          idx - 1] + datetime.timedelta(microseconds=1)
      count += 1
  rospy.loginfo("Fixed {} misaligned time frames.".format(count))
  return all_results


def label_proprioceptive_data(all_data, num_seconds_lookahead=3):
  """Label foot force std from robot trajectories."""
  timestamps = all_data['timestamps']

  # Clean and label data
  result_timestamp = []
  mean_steer_commands, mean_speed_commands, std_speed_commands = [], [], []
  mean_actual_speeds, std_actual_speeds, contact_ious = [], [], []
  powers, imu_rates, imus, image_embeddings = [], [], [], []
  for idx in tqdm(range(len(timestamps))):
    if (timestamps[-1] -
        timestamps[idx]).total_seconds() < num_seconds_lookahead:
      break
    furthest_idx = idx
    while (timestamps[furthest_idx] -
           timestamps[idx]).total_seconds() < num_seconds_lookahead:
      furthest_idx += 1

    mean_steering = np.mean(
        np.abs(all_data['steer_commands'][idx:furthest_idx]))
    curr_steering = np.abs(all_data['steer_commands'][idx])

    is_gait_consistent = (furthest_idx - idx > 200) and (
        mean_steering < 0.1) and (curr_steering < 0.1)
    if (not FLAGS.filter_out_inconsistent_gaits) or is_gait_consistent:
      result_timestamp.append(timestamps[idx])
      mean_actual_speeds.append(
          np.mean(all_data['actual_speeds'][idx:furthest_idx], axis=0))
      std_actual_speeds.append(
          np.std(all_data['actual_speeds'][idx:furthest_idx], axis=0))
      mean_speed_commands.append(
          np.mean(all_data['speed_commands'][idx:furthest_idx], axis=0))
      std_speed_commands.append(
          np.std(all_data['speed_commands'][idx:furthest_idx], axis=0))
      powers.append(np.mean(all_data['powers'][idx:furthest_idx]))
      imu_rates.append(
          np.mean(np.abs(all_data['imu_rates'][idx:furthest_idx]), axis=0))
      imus.append(np.mean(np.abs(all_data['imus'][idx:furthest_idx]), axis=0))
      mean_steer_commands.append(
          np.mean(np.abs(all_data['steer_commands'][idx:furthest_idx])))
      image_embeddings.append(all_data['image_embeddings'][idx])

  return dict(timestamp=np.array(result_timestamp),
              mean_speed_commands=np.array(mean_speed_commands),
              std_speed_commands=np.array(std_speed_commands),
              mean_actual_speeds=np.array(mean_actual_speeds),
              std_actual_speeds=np.array(std_actual_speeds),
              powers=np.array(powers),
              imu_rates=np.array(imu_rates),
              imus=np.array(imus),
              mean_steer_commands=np.array(mean_steer_commands),
              image_embeddings=np.array(image_embeddings),
              contact_ious=np.array(contact_ious))


def main(argv):
  del argv  # unused

  rospy.loginfo("Loading data from disk...")
  proprioceptive_data = load_proprioceptive_data(FLAGS.logdir)
  np.savez(os.path.join(FLAGS.output_dir, 'raw_proprioceptive_data.npz'),
           **proprioceptive_data)
  rospy.loginfo("Cleaning up and Labeling Data...")
  labeled_proprioceptive_data = label_proprioceptive_data(proprioceptive_data)

  np.savez(os.path.join(FLAGS.output_dir, 'labeled_proprioceptive_data.npz'),
           **labeled_proprioceptive_data)


if __name__ == "__main__":
  app.run(main)
