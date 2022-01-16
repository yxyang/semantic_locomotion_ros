"""Implements different metric for evaluating a trajectory."""
import numpy as np


def safety_metric(states):
  return states[-1]['timestamp'] - states[0]['timestamp']


def energy_metric(states):
  """Computes cost of transport"""
  timestamps = np.array([frame['timestamp'] for frame in states])
  motor_velocities = np.array([frame["motor_velocities"] for frame in states])
  motor_torques = np.array([frame["motor_torques"] for frame in states])

  energy = np.maximum(
      motor_torques * motor_velocities + 0.3 * motor_torques**2, 0)
  energy = np.sum(energy, axis=-1)

  total_energy = np.sum(energy[:-1] * np.diff(timestamps))
  total_time = np.maximum(timestamps[-1] - timestamps[0], 0.01)
  return total_energy / total_time


def stability_metric(states):
  base_rpy_rate = np.array([frame["base_rpy_rate"] for frame in states])
  return np.mean(np.square(base_rpy_rate))

def speed_metric(states):
  timestamps = np.array([frame['timestamp'] for frame in states])
  base_velocities = np.array([frame["base_velocity"] for frame in states])[:,
                                                                           0]
  total_distance = np.sum(base_velocities[:-1] * np.diff(timestamps))
  total_time = np.maximum(timestamps[-1] - timestamps[0], 0.01)
  return total_distance / total_time


def foot_velocity_metric(states):
  vel_stds = []
  for frame in states:
    foot_phase = np.fmod(frame['gait_generator_phase'] + 2 * np.pi, 2 * np.pi)
    foot_contact = np.logical_and(foot_phase > 1, foot_phase < np.pi - 1)
    if np.sum(foot_contact):
      contact_vels = frame['foot_velocities'][np.nonzero(foot_contact)[0]]
      vel_stds.append(np.std(contact_vels, axis=0))

  return np.nan_to_num(np.mean(vel_stds), 0)


def foot_force_metric(states):
  force_stds = []
  for frame in states:
    if np.sum(frame['foot_contacts']):
      contact_forces = frame['foot_forces'][np.nonzero(
          frame['foot_contacts'])[0]]
      force_stds.append(np.std(contact_forces, axis=0))
  return np.mean(force_stds)
