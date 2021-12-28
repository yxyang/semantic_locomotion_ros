#!/usr/bin/env python
"""Example of convex MPCcontroller on A1 robot."""
from absl import app
from absl import flags

import rospy

from a1_interface.msg import controller_mode
from a1_interface.msg import gait_type
from a1_interface.msg import robot_state
from a1_interface.msg import speed_command
from a1_interface.convex_mpc_controller import locomotion_controller

flags.DEFINE_string("logdir", None, "where to log trajectories.")
flags.DEFINE_bool("use_real_robot", False,
                  "whether to use real robot or simulation")
flags.DEFINE_bool("show_gui", False, "whether to show GUI.")
FLAGS = flags.FLAGS

WATCHDOG_TIMEOUT_SECS = 1


def main(argv):
  del argv  # unused
  controller = locomotion_controller.LocomotionController(
      FLAGS.use_real_robot, FLAGS.show_gui, FLAGS.logdir)

  rospy.Subscriber("controller_mode", controller_mode,
                   controller.set_controller_mode)
  rospy.Subscriber("speed_command", speed_command,
                   controller.set_desired_speed)
  rospy.Subscriber("gait_type", gait_type, controller.set_gait)
  robot_state_publisher = rospy.Publisher('robot_state',
                                          robot_state,
                                          queue_size=10)
  rospy.init_node("a1_interface", anonymous=True)

  rate = rospy.Rate(20)
  while not rospy.is_shutdown():
    robot = controller.robot
    state = robot_state(is_safe=controller.is_safe,
                        controller_mode=controller.mode,
                        gait_type=controller.gait,
                        timestamp=rospy.get_rostime(),
                        base_velocity=robot.base_velocity,
                        base_orientation_rpy=robot.base_orientation_rpy,
                        motor_angles=robot.motor_angles,
                        motor_velocities=robot.motor_velocities,
                        motor_torques=robot.motor_torques,
                        foot_contacts=robot.foot_contacts)
    robot_state_publisher.publish(state)
    if controller.time_since_reset - controller.last_command_timestamp \
      > WATCHDOG_TIMEOUT_SECS:
      rospy.loginfo("Controller node timeout, stopping robot.")
      controller.set_controller_mode(
          controller_mode(mode=controller_mode.DOWN,
                          timestamp=rospy.get_rostime()))

    if not controller.is_safe:
      rospy.loginfo("Robot unsafe, stopping robot.")
      controller.set_controller_mode(
          controller_mode(mode=controller_mode.DOWN,
                          timestamp=rospy.get_rostime()))
    rate.sleep()

  controller.set_controller_mode(
      controller_mode(mode=controller_mode.TERMINATE,
                      timestamp=rospy.get_rostime()))


if __name__ == "__main__":
  app.run(main)
