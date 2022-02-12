#!/usr/bin/env python
"""Generates energy efficient timing coef based on given speed."""
from absl import app
from absl import flags

import numpy as np
import rospy

from a1_interface.msg import robot_state, speed_command, gait_command
from simple_gait_optimization.timing_policy.timing_policy import TimingPolicy

FLAGS = flags.FLAGS


def get_max_forward_speed(step_freq, max_swing_distance=0.3):
  return 2 * step_freq * max_swing_distance


def convert_to_linear_speed_equivalent(vx, vy, wz):
  return np.sqrt(vx**2 + 4 * vy**2) + np.abs(wz)


class StateListener:
  """Records desired and actual speed from subscribed topics."""
  def __init__(self):
    self._current_speed = 0.
    self._desired_speed = 0.

  def record_current_speed(self, robot_state_msg):
    self._current_speed = convert_to_linear_speed_equivalent(
        robot_state_msg.base_velocity[0], robot_state_msg.base_velocity[1], 0)

  def record_desired_speed(self, speed_command_msg):
    self._desired_speed = convert_to_linear_speed_equivalent(
        speed_command_msg.vel_x, speed_command_msg.vel_y,
        speed_command_msg.rot_z)

  @property
  def current_speed(self):
    return np.array([self._current_speed, 0, 0])

  @property
  def desired_speed(self):
    return np.array([self._desired_speed, 0, 0, 0])


def main(argv):
  del argv  # unused
  rospy.init_node('simple_gait_optimization', anonymous=True)

  state_listener = StateListener()
  rospy.Subscriber('speed_command', speed_command,
                   state_listener.record_desired_speed)
  rospy.Subscriber('robot_state', robot_state,
                   state_listener.record_current_speed)
  gait_command_publisher = rospy.Publisher('autogait_command',
                                           gait_command,
                                           queue_size=1)

  policy = TimingPolicy()
  rate = rospy.Rate(20)
  while not rospy.is_shutdown():
    timing_parameters = policy.get_action(state_listener.current_speed,
                                          state_listener.desired_speed)
    desired_gait = gait_command(timing_parameters=list(timing_parameters),
                                foot_clearance=0.13,
                                base_height=0.26,
                                max_forward_speed=get_max_forward_speed(
                                    timing_parameters[0]),
                                timestamp=rospy.get_rostime())
    gait_command_publisher.publish(desired_gait)

    rate.sleep()


if __name__ == "__main__":
  app.run(main)
