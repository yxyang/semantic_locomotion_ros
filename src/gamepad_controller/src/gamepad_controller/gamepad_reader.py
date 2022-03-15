#!/usr/bin/env python
"""Gamepad reader example."""
from absl import app
from absl import flags
import rospy
from std_msgs.msg import Bool, String  #, Float32

import numpy as np

from a1_interface.msg import controller_mode
from a1_interface.msg import gait_command
from a1_interface.msg import robot_state
from a1_interface.msg import speed_command

from gamepad_controller.gamepad_reader_lib import Gamepad, GaitMode

FLAGS = flags.FLAGS


class RobotStateListener:
  """Simple class to capture latest robot state."""
  def __init__(self):
    self._is_safe = False
    self._controller_mode = controller_mode.DOWN

  def callback(self, msg):
    self._is_safe = msg.is_safe
    self._controller_mode = msg.controller_mode

  @property
  def is_safe(self):
    return self._is_safe

  @property
  def controller_mode(self):
    return self._controller_mode


class GaitCommandListener:
  """Listens and stores auto-gait command."""
  def __init__(self):
    self._desired_gait_type = gait_command(
        timing_parameters=[3.5, np.pi, np.pi, 0, 0.5],
        base_height=0.26,
        foot_clearance=0.13,
        timestamp=rospy.get_rostime())

  def callback(self, msg):
    self._desired_gait_type = msg

  @property
  def desired_gait_type(self):
    return self._desired_gait_type


class SpeedCommandListener:
  """Listens and stores auto-speed command."""
  def __init__(self):
    self._desired_speed = speed_command(vel_x=0,
                                        vel_y=0,
                                        rot_z=0,
                                        timestamp=rospy.get_rostime())

  def callback(self, msg):
    self._desired_speed = msg

  @property
  def desired_speed(self):
    return self._desired_speed


def main(_):
  rospy.init_node('gamepad_controller', anonymous=True)
  gamepad = Gamepad()
  controller_mode_publisher = rospy.Publisher('controller_mode',
                                              controller_mode,
                                              queue_size=1)
  speed_command_publisher = rospy.Publisher('speed_command',
                                            speed_command,
                                            queue_size=1)
  estop_publisher = rospy.Publisher('estop', Bool, queue_size=1)
  autogait_publisher = rospy.Publisher('autogait', String, queue_size=1)

  # Define listeners
  robot_state_listener = RobotStateListener()
  rospy.Subscriber('robot_state', robot_state, robot_state_listener.callback)
  gait_command_listener = GaitCommandListener()
  rospy.Subscriber('autogait_command', gait_command,
                   gait_command_listener.callback)
  speed_command_listener = SpeedCommandListener()
  rospy.Subscriber('autospeed_command', speed_command,
                   speed_command_listener.callback)

  gait_command_publisher = rospy.Publisher('gait_command',
                                           gait_command,
                                           queue_size=1)

  rate = rospy.Rate(20)
  while not rospy.is_shutdown():
    if not robot_state_listener.is_safe and not gamepad.estop_flagged:
      rospy.loginfo('Estop automatically flagged.')
      gamepad.flag_estop()
    controller_mode_publisher.publish(gamepad.mode_command)
    estop_publisher.publish(gamepad.estop_flagged)
    autogait_publisher.publish(str(gamepad.gait_mode))
    if gamepad.gait_mode == GaitMode.MANUAL_SPEED_MANUAL_GAIT:
      gait_command_publisher.publish(gamepad.gait_command)
      speed_command_publisher.publish(gamepad.speed_command)
    elif gamepad.gait_mode == GaitMode.MANUAL_SPEED_AUTO_GAIT:
      desired_gait = gait_command_listener.desired_gait_type
      gait_command_publisher.publish(desired_gait)
      cmd = gamepad.speed_command
      speed_command_publisher.publish(cmd)
    else:
      # AUTO_SPEED_AUTO_GAIT, human could adjust speed
      desired_gait = gait_command_listener.desired_gait_type
      gait_command_publisher.publish(desired_gait)
      cmd = gamepad.speed_command
      neutral_speed = speed_command_listener.desired_speed.vel_x
      cmd.vel_x /= gamepad.vel_scale_x
      cmd.vel_x = np.where(
          cmd.vel_x < 0,
          # Brake
          (cmd.vel_x + 1) * neutral_speed,
          # Accelerate
          neutral_speed + cmd.vel_x *
          (gamepad.vel_scale_x - neutral_speed))
      cmd.vel_y = gamepad.speed_command.vel_y
      cmd.rot_z = gamepad.speed_command.rot_z
      speed_command_publisher.publish(cmd)
    rate.sleep()

  gamepad.stop()


if __name__ == "__main__":
  app.run(main)
