"""Implements different metric for evaluating a trajectory."""
import numpy as np


def safety_metric(states):
  return len(states)


def energy_metric(states):
  """Computes cost of transport"""
  motor_velocities = np.array([frame["motor_velocities"] for frame in states])
  motor_torques = np.array([frame["motor_torques"] for frame in states])

  energy = np.maximum(
      motor_torques * motor_velocities + 0.3 * motor_torques**2, 0)
  energy = np.sum(energy, axis=-1)
  return np.mean(energy)


def stability_metric(states):
  base_rpy_rate = np.array([frame["base_rpy_rate"] for frame in states])
  return np.mean(np.square(base_rpy_rate))


def speed_metric(states):
  base_velocities = np.array([frame["base_velocity"] for frame in states])[:,
                                                                           0]
  return np.mean(base_velocities)
