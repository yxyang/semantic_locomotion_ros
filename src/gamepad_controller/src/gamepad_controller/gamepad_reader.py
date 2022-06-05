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
from m1_perception.gait_policy.manual_gait_policy import ManualGaitPolicy

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
  gaitmode_publisher = rospy.Publisher('gait_mode', String, queue_size=1)

  # Define listeners
  robot_state_listener = RobotStateListener()
  rospy.Subscriber('robot_state', robot_state, robot_state_listener.callback)
  speed_command_listener = SpeedCommandListener()
  rospy.Subscriber('autospeed_command', speed_command,
                   speed_command_listener.callback)
  nav_command_listener = SpeedCommandListener()
  rospy.Subscriber('/navigation/speed_command', speed_command,
                   nav_command_listener.callback)

  gait_policy = ManualGaitPolicy()
  gait_command_publisher = rospy.Publisher('gait_command',
                                           gait_command,
                                           queue_size=1)
  skip_waypoint_publisher = rospy.Publisher('/skip_waypoint',
                                            Bool,
                                            queue_size=1)

  rate = rospy.Rate(20)
  while not rospy.is_shutdown():
    if not robot_state_listener.is_safe and not gamepad.estop_flagged:
      rospy.loginfo('Estop automatically flagged.')
      gamepad.flag_estop()
    controller_mode_publisher.publish(gamepad.mode_command)
    estop_publisher.publish(gamepad.estop_flagged)
    gaitmode_publisher.publish(str(gamepad.gait_mode))

    if gamepad.skip_waypoint:
      rospy.loginfo("Skipping waypoint.")
      skip_waypoint_publisher.publish(True)
      gamepad.skip_waypoint = False

    if gamepad.gait_mode == GaitMode.MANUAL_SPEED_MANUAL_GAIT:
      gait_command_publisher.publish(gamepad.gait_command)
      speed_command_publisher.publish(gamepad.speed_command)
    elif gamepad.gait_mode == GaitMode.MANUAL_SPEED_AUTO_GAIT:
      desired_gait = gait_policy.get_action(gamepad.speed_command)
      gait_command_publisher.publish(desired_gait)
      cmd = gamepad.speed_command
      speed_command_publisher.publish(cmd)
    elif gamepad.gait_mode == GaitMode.AUTO_SPEED_AUTO_GAIT:
      # AUTO_SPEED_AUTO_GAIT, human could adjust speed
      cmd = gamepad.speed_command
      neutral_x = speed_command_listener.desired_speed.vel_x
      cmd.vel_x /= gamepad.vel_scale_x
      cmd.vel_x = np.where(
          cmd.vel_x < 0,
          # Brake
          -0.5 + (cmd.vel_x + 1) * (neutral_x + 0.5),
          # Accelerate
          neutral_x + cmd.vel_x * (gamepad.vel_scale_x - neutral_x))
      neutral_y = speed_command_listener.desired_speed.vel_y
      cmd.vel_y /= gamepad.vel_scale_y  # Normalize to [-1, 1]
      cmd.vel_y = np.where(
          cmd.vel_y < 0, -gamepad.vel_scale_y + (cmd.vel_y + 1) *
          (neutral_y + gamepad.vel_scale_y),
          neutral_y + cmd.vel_y * (gamepad.vel_scale_y - neutral_y))
      cmd.rot_z = gamepad.speed_command.rot_z
      speed_command_publisher.publish(cmd)
      desired_gait = gait_policy.get_action(cmd)
      gait_command_publisher.publish(desired_gait)
    else:
      # AutoNav
      cmd = gamepad.speed_command
      neutral_x = nav_command_listener.desired_speed.vel_x
      cmd.vel_x /= gamepad.vel_scale_x
      cmd.vel_x = np.where(
          cmd.vel_x < 0,
          # Brake
          -0.5 + (cmd.vel_x + 1) * (neutral_x + 0.5),
          # Accelerate
          neutral_x + cmd.vel_x * (gamepad.vel_scale_x - neutral_x))

      neutral_y = nav_command_listener.desired_speed.vel_y
      cmd.vel_y /= gamepad.vel_scale_y  # Normalize to [-1, 1]
      cmd.vel_y = np.where(
          cmd.vel_y < 0, -gamepad.vel_scale_y + (cmd.vel_y + 1) *
          (neutral_y + gamepad.vel_scale_y),
          neutral_y + cmd.vel_y * (gamepad.vel_scale_y - neutral_y))

      neutral_rot = nav_command_listener.desired_speed.rot_z
      cmd.rot_z /= gamepad.vel_scale_rot
      cmd.rot_z = np.where(
          cmd.rot_z < 0, -gamepad.vel_scale_rot + (cmd.rot_z + 1) *
          (neutral_rot + gamepad.vel_scale_rot),
          neutral_rot + cmd.rot_z * (gamepad.vel_scale_rot - neutral_rot))

      speed_command_publisher.publish(cmd)
      desired_gait = gait_policy.get_action(cmd)
      gait_command_publisher.publish(desired_gait)

    rate.sleep()

  gamepad.stop()


if __name__ == "__main__":
  app.run(main)
