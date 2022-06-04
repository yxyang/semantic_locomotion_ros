#!/usr/bin/env python
"""Generates gait timing from desired and actual speed."""

import numpy as np
import rospy

from a1_interface.msg import gait_command


def convert_to_linear_speed_equivalent(vx, vy, wz):
  return np.sqrt(vx**2 + 4 * vy**2) + np.abs(wz)


class ManualGaitPolicy:
  """Outputs timing commands based on desired and actual speed of the robot."""
  def __init__(self, min_freq=2.5, max_freq=3.8, max_speed=2):
    self._min_freq = min_freq
    self._max_freq = max_freq
    self._max_speed = max_speed

  def get_action(self, speed_command_msg):
    desired_vel = convert_to_linear_speed_equivalent(speed_command_msg.vel_x,
                                                     speed_command_msg.vel_y,
                                                     speed_command_msg.rot_z)
    desired_vel = np.clip(desired_vel, 0, self._max_speed)
    step_freq = self._min_freq + (
        self._max_freq - self._min_freq) * desired_vel / self._max_speed
    return gait_command(
        timing_parameters=[step_freq, np.pi, np.pi, 0, 0.5],
        foot_clearance=0.18 - 0.08 * desired_vel / self._max_speed,
        base_height=0.31 -
        0.05 * np.clip(2 * desired_vel / self._max_speed, 0, 1),
        timestamp=rospy.get_rostime())
