#!/usr/bin/env python
"""Example of convex MPCcontroller on A1 robot."""
from absl import app
from absl import flags

from std_msgs.msg import Bool
from std_msgs.msg import String
import rospy

from a1_interface.msg import controller_mode
from a1_interface.msg import gait_type
from a1_interface.msg import robot_state
from a1_interface.msg import speed_command

CONTROLLER_MODE_MAP = {0: "Down", 1: "Stand", 2: "Walk", 3: "Terminate"}
GAIT_TYPE_MAP = {0: "Slow", 1: "Medium", 2: "Fast"}


class RobotStateRecorder:
  def __init__(self):
    self._robot_state = robot_state()

  def record_state(self, robot_state):
    self._robot_state = robot_state

  def get_info_string(self):
    return "Controller: {}\nGait: {}\n".format(
        CONTROLLER_MODE_MAP[self._robot_state.controller_mode],
        GAIT_TYPE_MAP[self._robot_state.gait_type])


class ControllerStateRecorder:
  def __init__(self):
    self._estop = False

  def record_estop(self, estop):
    self._estop = estop.data

  def get_info_string(self):
    if self._estop:
      return "ESTOP!"
    else:
      return


class AutogaitRecorder:
  def __init__(self):
    self._autogait = False

  def record_autogait(self, autogait):
    self._autogait = autogait.data

  def get_info_string(self):
    if self._autogait:
      return "Auto Gait"
    else:
      return "Manual Gait"


def main(argv):
  del argv  # unused
  state_recorder = RobotStateRecorder()
  controller_recorder = ControllerStateRecorder()
  autogait_recorder = AutogaitRecorder()
  rospy.Subscriber("robot_state", robot_state, state_recorder.record_state)
  rospy.Subscriber("estop", Bool, controller_recorder.record_estop)
  rospy.Subscriber("autogait", Bool, autogait_recorder.record_autogait)
  robot_state_publisher = rospy.Publisher('status/robot_state_string',
                                          String,
                                          queue_size=1)
  controller_state_publisher = rospy.Publisher('status/estop_string',
                                               String,
                                               queue_size=1)
  rospy.init_node("info_publisher", anonymous=True)

  rate = rospy.Rate(20)
  while not rospy.is_shutdown():
    robot_state_publisher.publish(state_recorder.get_info_string() +
                                  autogait_recorder.get_info_string())
    controller_state_publisher.publish(controller_recorder.get_info_string())
    rate.sleep()


if __name__ == "__main__":
  app.run(main)
