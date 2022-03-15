#!/usr/bin/env python
"""Generates gait timing from desired and actual speed."""

import numpy as np
import rospy

from a1_interface.msg import gait_command


def get_max_forward_speed(step_freq, max_swing_distance=0.3):
  return 2 * step_freq * max_swing_distance


class ManualGaitPolicy:
  """Outputs timing commands based on desired and actual speed of the robot."""
  def __init__(self, min_freq=2.5, max_freq=3.5, max_speed=2):
    self._min_freq = min_freq
    self._max_freq = max_freq
    self._max_speed = max_speed

  def get_action(self, desired_vel):
    desired_vel = np.clip(desired_vel, 0, self._max_speed)
    step_freq = self._min_freq + (
        self._max_freq - self._min_freq) * desired_vel / self._max_speed
    return gait_command(
        timing_parameters=[step_freq, np.pi, np.pi, 0, 0.5],
        foot_clearance=0.18 - 0.08 * desired_vel / self._max_speed,
        base_height=0.31 - 0.05 * desired_vel / self._max_speed,
        max_forward_speed=get_max_forward_speed(step_freq),
        recommended_forward_speed=0,
        timestamp=rospy.get_rostime())
