#!/usr/bin/env python
"""Generates energy efficient timing coef based on given speed."""
from absl import app
from absl import flags

import numpy as np
import rospy

from a1_interface.msg import speed_command, gait_command
from m1_perception.gait_policy.manual_gait_policy import ManualGaitPolicy

FLAGS = flags.FLAGS


def convert_to_linear_speed_equivalent(vx, vy, wz):
  return np.sqrt(vx**2 + 4 * vy**2) + np.abs(wz)


class StateListener:
  """Records desired and actual speed from subscribed topics."""
  def __init__(self):
    self._desired_speed = 0.

  def record_desired_speed(self, speed_command_msg):
    self._desired_speed = convert_to_linear_speed_equivalent(
        speed_command_msg.vel_x, speed_command_msg.vel_y,
        speed_command_msg.rot_z)

  @property
  def desired_speed(self):
    return self._desired_speed


def main(argv):
  del argv  # unused
  rospy.init_node('gait_from_speed', anonymous=True)

  state_listener = StateListener()
  rospy.Subscriber('speed_command', speed_command,
                   state_listener.record_desired_speed)
  gait_command_publisher = rospy.Publisher('autogait_command',
                                           gait_command,
                                           queue_size=1)

  policy = ManualGaitPolicy()
  rate = rospy.Rate(20)
  while not rospy.is_shutdown():
    desired_gait = policy.get_action(state_listener.desired_speed)
    gait_command_publisher.publish(desired_gait)

    rate.sleep()


if __name__ == "__main__":
  app.run(main)
