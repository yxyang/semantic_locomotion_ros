#!/usr/bin/env python
"""Gamepad reader example."""
from absl import app
from absl import flags
import rospy
from std_msgs.msg import Bool

from a1_interface.msg import controller_mode
from a1_interface.msg import gait_type
from a1_interface.msg import robot_state
from a1_interface.msg import speed_command
from gamepad_reader_lib import Gamepad

flags.DEFINE_bool('publish_gaits', True, 'whether to publish gait commands.')
FLAGS = flags.FLAGS


class RobotStateListener:
  """Simple class to capture latest robot state."""
  def __init__(self):
    self._is_safe = False
    self._controller_mode = controller_mode.DOWN

  def callback(self, data):
    self._is_safe = data.is_safe
    self._controller_mode = data.controller_mode

  @property
  def is_safe(self):
    return self._is_safe

  @property
  def controller_mode(self):
    return self._controller_mode


def main(_):
  gamepad = Gamepad()
  controller_mode_publisher = rospy.Publisher('controller_mode',
                                              controller_mode,
                                              queue_size=1)
  speed_command_publisher = rospy.Publisher('speed_command',
                                            speed_command,
                                            queue_size=1)
  estop_publisher = rospy.Publisher('estop', Bool, queue_size=1)
  robot_state_listener = RobotStateListener()
  rospy.Subscriber('robot_state', robot_state, robot_state_listener.callback)
  if FLAGS.publish_gaits:
    gait_type_publisher = rospy.Publisher('gait_type', gait_type, queue_size=1)
  rospy.init_node('gamepad_controller', anonymous=True)

  rate = rospy.Rate(20)
  while not rospy.is_shutdown():
    if not robot_state_listener.is_safe and not gamepad.estop_flagged:
      rospy.loginfo('Estop automatically flagged.')
      gamepad.flag_estop()
    controller_mode_publisher.publish(gamepad.mode_command)
    speed_command_publisher.publish(gamepad.speed_command)
    estop_publisher.publish(gamepad.estop_flagged)
    if FLAGS.publish_gaits:
      gait_type_publisher.publish(gamepad.gait_command)
    rate.sleep()

  gamepad.stop()


if __name__ == "__main__":
  app.run(main)
