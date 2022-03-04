#!/usr/bin/env python
"""Generates gait timing from desired and actual speed."""

from absl import app

import numpy as np


class ManualTimingPolicy:
  """Outputs timing commands based on desired and actual speed of the robot."""
  def __init__(self, min_freq=2.5, max_freq=3.5, max_speed=2):
    self._min_freq = min_freq
    self._max_freq = max_freq
    self._max_speed = max_speed

  def get_action(self, base_vel, desired_vel):
    del base_vel  # unused
    desired_vel = np.clip(desired_vel[0], 0, self._max_speed)
    step_freq = self._min_freq + (
        self._max_freq - self._min_freq) * desired_vel / self._max_speed
    return np.array([step_freq, np.pi, np.pi, 0, 0.5])


def main(_):
  policy = ManualTimingPolicy()
  base_vel = np.zeros(3)
  desired_vel = np.zeros(4)
  print(policy.get_action(base_vel, desired_vel))


if __name__ == "__main__":
  app.run(main)
