#!/usr/bin/env python
"""Subscribes to robot state and publish as TF transforms."""

from absl import app
from absl import flags

import geometry_msgs
import rospy
import tf2_ros

import pybullet as p

from a1_interface.msg import robot_state

FLAGS = flags.FLAGS

CAMERA_POSITION_ROBOT_FRAME = [0.24, 0, 0]
CAMERA_ORIENTATION_TRANFORM_QUAT = [0.5, -0.5, 0.5, -0.5]


class TFPublisher:
  """Publish robot state as TF events."""
  def __init__(self):
    self.broadcaster = tf2_ros.TransformBroadcaster()
    self._robot_orientation_quat_xyzw = [0, 0, 0, 1]

  def robot_state_callback(self, robot_state_data):
    """Publish robot state data to TF."""
    transform_msg = geometry_msgs.msg.TransformStamped()
    transform_msg.header.stamp = robot_state_data.timestamp
    transform_msg.header.frame_id = "world"
    transform_msg.child_frame_id = "base_link"
    transform_msg.transform.translation.x = 0.
    transform_msg.transform.translation.y = 0.
    transform_msg.transform.translation.z = robot_state_data.base_position[2]
    q = robot_state_data.base_orientation_quat_xyzw
    self._robot_orientation_quat_xyzw = q
    transform_msg.transform.rotation.x = q[0]
    transform_msg.transform.rotation.y = q[1]
    transform_msg.transform.rotation.z = q[2]
    transform_msg.transform.rotation.w = q[3]
    self.broadcaster.sendTransform(transform_msg)

    transform_msg = geometry_msgs.msg.TransformStamped()
    transform_msg.header.stamp = robot_state_data.timestamp
    transform_msg.header.frame_id = "base_link"
    transform_msg.child_frame_id = "base_footprint"
    transform_msg.transform.translation.x = 0.
    transform_msg.transform.translation.y = 0.
    base_height = robot_state_data.base_position_ground_frame[
        2]
    transform_msg.transform.translation.z = -base_height
    _, q = p.invertTransform(
        [0, 0, 0], robot_state_data.base_orientation_ground_frame_quat_xyzw)
    transform_msg.transform.rotation.x = q[0]
    transform_msg.transform.rotation.y = q[1]
    transform_msg.transform.rotation.z = q[2]
    transform_msg.transform.rotation.w = q[3]
    self.broadcaster.sendTransform(transform_msg)

    transform_msg = geometry_msgs.msg.TransformStamped()
    transform_msg.header.stamp = robot_state_data.timestamp
    transform_msg.header.frame_id = "base_link"
    transform_msg.child_frame_id = "camera_link"
    transform_msg.transform.translation.x = CAMERA_POSITION_ROBOT_FRAME[0]
    transform_msg.transform.translation.y = CAMERA_POSITION_ROBOT_FRAME[1]
    transform_msg.transform.translation.z = CAMERA_POSITION_ROBOT_FRAME[2]
    transform_msg.transform.rotation.x = CAMERA_ORIENTATION_TRANFORM_QUAT[0]
    transform_msg.transform.rotation.y = CAMERA_ORIENTATION_TRANFORM_QUAT[1]
    transform_msg.transform.rotation.z = CAMERA_ORIENTATION_TRANFORM_QUAT[2]
    transform_msg.transform.rotation.w = CAMERA_ORIENTATION_TRANFORM_QUAT[3]
    self.broadcaster.sendTransform(transform_msg)


def main(argv):
  del argv  # unused
  rospy.init_node('tf_publisher', anonymous=True)

  publisher = TFPublisher()
  rospy.Subscriber("/robot_state", robot_state, publisher.robot_state_callback)
  rospy.spin()


if __name__ == "__main__":
  app.run(main)
