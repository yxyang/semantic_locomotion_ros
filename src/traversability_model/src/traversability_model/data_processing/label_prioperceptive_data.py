#!/usr/bin/env python
"""Generates training data from prioperceptive and perception data."""
import datetime
import multiprocessing
import os
import pickle

from absl import app
from absl import flags

import numpy as np
import rospy
from tqdm import tqdm

flags.DEFINE_string('logdir', None, 'directory for prioperceptive data.')
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
  gait_types = []
  timestamps = []
  foot_forces = []
  foot_phases = []
  actual_speeds = []
  powers = []
  imu_rates = []
  robot_states = pickle.load(open(os.path.join(base_dir, filename), 'rb'))
  for frame in robot_states[:-1]:
    foot_phases.append(frame['gait_generator_phase'])
    speed_commands.append(frame['desired_speed'][0])
    steer_commands.append(frame['desired_speed'][1])
    foot_forces.append(frame['foot_forces'])
    gait_types.append(frame['gait_type'])
    actual_speeds.append(frame['base_vels_body_frame'])
    power = np.maximum(
        frame['motor_torques'] * frame['motor_vels'] +
        0.3 * frame['motor_torques']**2, 0)
    powers.append(power)
    imu_rates.append(np.sum(np.square(frame['base_rpy_rate'][:2])))

    delta_time_s = datetime.timedelta(seconds=frame['timestamp'] -
                                      robot_states[-1]['timestamp'])
    timestamps.append(end_ts + delta_time_s)
  return dict(foot_phases=foot_phases,
              speed_commands=speed_commands,
              steer_commands=steer_commands,
              gait_types=gait_types,
              timestamps=timestamps,
              foot_forces=foot_forces,
              actual_speeds=actual_speeds,
              powers=powers,
              imu_rates=imu_rates)


def load_prioperceptive_data(logdir):
  """Loads prioperceptive data.

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
    all_results[key] = np.concatenate([result[key] for result in results],
                                      axis=0)
  return all_results


def extract_steps_and_max_stance_forces(timestamp, foot_phase, foot_force):
  """Extracts steps taken and maximum contact force in each step."""
  idx = 0
  foot_phase = np.fmod(foot_phase + 2 * np.pi, 2 * np.pi)
  step_indices, step_times, step_forces = [], [], []
  pbar = tqdm(total=len(timestamp))
  while True:
    while foot_phase[idx] < np.pi:
      idx += 1
      pbar.update(1)
      if idx == len(timestamp):
        return step_indices, step_times, step_forces
    start_idx = idx

    while foot_phase[idx] >= np.pi:
      idx += 1
      pbar.update(1)
      if idx == len(timestamp):
        return step_indices, step_times, step_forces
    end_idx = idx

    max_idx = np.argmax(foot_force[start_idx:end_idx]) + start_idx
    step_indices.append(max_idx)
    step_times.append(timestamp[max_idx])
    step_forces.append(foot_force[max_idx])


def label_prioperceptive_data(all_data, num_steps_lookahead=10):
  """Label foot force std from robot trajectories."""
  # Get maximum contact force for each step
  timestamps = all_data['timestamps']
  foot_phases = all_data['foot_phases']
  foot_forces = all_data['foot_forces']
  gait_types = all_data['gait_types']
  fr_indices, fr_times, fr_forces = extract_steps_and_max_stance_forces(
      timestamps, foot_phases[:, 0], foot_forces[:, 0])
  fl_indices, fl_times, fl_forces = extract_steps_and_max_stance_forces(
      timestamps, foot_phases[:, 1], foot_forces[:, 1])
  rr_indices, rr_times, rr_forces = extract_steps_and_max_stance_forces(
      timestamps, foot_phases[:, 2], foot_forces[:, 2])
  rl_indices, rl_times, rl_forces = extract_steps_and_max_stance_forces(
      timestamps, foot_phases[:, 3], foot_forces[:, 3])

  step_indices = [fr_indices, fl_indices, rr_indices, rl_indices]
  step_times = [fr_times, fl_times, rr_times, rl_times]
  step_forces = [fr_forces, fl_forces, rr_forces, rl_forces]

  # Clean and label data
  result_timestamp, result_diffs, result_gaits = [], [], []
  speed_commands, actual_speeds, powers, imu_rates = [], [], [], []
  pointers = [0, 0, 0, 0]
  for idx in tqdm(range(len(timestamps))):
    force_diffs = []
    furthest_idx = idx
    for leg_id in range(4):
      while step_times[leg_id][pointers[leg_id]] < timestamps[idx]:
        pointers[leg_id] += 1
        if pointers[leg_id] + num_steps_lookahead >= len(step_times[leg_id]):
          return dict(timestamp=np.array(result_timestamp),
                      foot_force_difference=np.array(result_diffs),
                      gaits=np.array(result_gaits),
                      speed_commands=np.array(speed_commands),
                      actual_speeds=np.array(actual_speeds),
                      powers=np.array(powers),
                      imu_rates=imu_rates)
      force_diffs.append(
          np.std(step_forces[leg_id][pointers[leg_id]:pointers[leg_id] +
                                     num_steps_lookahead]))

      furthest_idx = np.maximum(
          furthest_idx,
          step_indices[leg_id][pointers[leg_id] + num_steps_lookahead])

    if FLAGS.filter_out_inconsistent_gaits:
      if (gait_types[idx:furthest_idx] == gait_types[idx]).all():
        result_timestamp.append(timestamps[idx])
        result_diffs.append(np.mean(force_diffs))
        result_gaits.append(gait_types[idx])
        actual_speeds.append(
            np.mean(all_data['actual_speeds'][idx:furthest_idx], axis=0))
        speed_commands.append(
            np.mean(all_data['speed_commands'][idx:furthest_idx], axis=0))
        powers.append(np.mean(all_data['powers'][idx:furthest_idx]))
        imu_rates.append(np.mean(all_data['imu_rates'][idx:furthest_idx]))
    else:
      result_timestamp.append(timestamps[idx])
      result_diffs.append(np.mean(force_diffs))
      result_gaits.append(gait_types[idx])
      actual_speeds.append(
          np.mean(all_data['actual_speeds'][idx:furthest_idx], axis=0))
      speed_commands.append(
          np.mean(all_data['speed_commands'][idx:furthest_idx], axis=0))
      powers.append(np.mean(all_data['powers'][idx:furthest_idx]))
      imu_rates.append(np.mean(all_data['imu_rates'][idx:furthest_idx]))


def main(argv):
  del argv  # unused

  rospy.loginfo("Loading data from disk...")
  prioperceptive_data = load_prioperceptive_data(FLAGS.logdir)
  rospy.loginfo("Cleaning up and Labeling Data...")
  labeled_prioperceptive_data = label_prioperceptive_data(prioperceptive_data)

  np.savez(os.path.join(FLAGS.output_dir, 'labeled_prioperceptive_data.npz'),
           **labeled_prioperceptive_data)


if __name__ == "__main__":
  app.run(main)
