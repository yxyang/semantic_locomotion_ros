#!/usr/bin/env python
"""Subscribes to robot state and publish as TF transforms."""

from absl import app
from absl import flags

import geometry_msgs.msg
from sensor_msgs.msg import Imu
import rospy
import tf2_ros

import pybullet as p

from a1_interface.msg import robot_state

FLAGS = flags.FLAGS

CAMERA_POSITION_ROBOT_FRAME = [0.24, 0, 0]
CAMERA_ORIENTATION_TRANFORM_QUAT = [0.5, -0.5, 0.5, 0.5]


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

  def camera_imu_callback(self, camera_imu):
    """Publish camera orientation data to TF."""
    camera_orientation = camera_imu.orientation
    camera_orientation_world_frame = [
        camera_orientation.x, camera_orientation.y, camera_orientation.z,
        camera_orientation.w
    ]
    camera_orientation_rpy = list(
        p.getEulerFromQuaternion(camera_orientation_world_frame))
    # Transform between camera and robot coordinate systems
    _, camera_orientation_world_frame = p.multiplyTransforms(
        [0., 0., 0.],
        camera_orientation_world_frame,
        [0., 0., 0.],
        CAMERA_ORIENTATION_TRANFORM_QUAT,
    )

    camera_orientation_rpy = list(
        p.getEulerFromQuaternion(camera_orientation_world_frame))
    robot_orientation_rpy = p.getEulerFromQuaternion(
        self._robot_orientation_quat_xyzw)
    # Align camera rpy and robot rpy
    camera_orientation_rpy[2] = robot_orientation_rpy[2]
    camera_orientation_world_frame = p.getQuaternionFromEuler(
        camera_orientation_rpy)
    robot_orientation_world_frame = self._robot_orientation_quat_xyzw

    _, world_orientation_robot_frame = p.invertTransform(
        [0., 0., 0.], robot_orientation_world_frame)
    _, camera_orientation_robot_frame = p.multiplyTransforms(
        [0., 0., 0.], world_orientation_robot_frame, [0., 0., 0.],
        camera_orientation_world_frame)

    transform_msg = geometry_msgs.msg.TransformStamped()
    transform_msg.header.stamp = camera_imu.header.stamp
    transform_msg.header.frame_id = "base_link"
    transform_msg.child_frame_id = "camera_link"
    transform_msg.transform.translation.x = CAMERA_POSITION_ROBOT_FRAME[0]
    transform_msg.transform.translation.y = CAMERA_POSITION_ROBOT_FRAME[1]
    transform_msg.transform.translation.z = CAMERA_POSITION_ROBOT_FRAME[2]
    transform_msg.transform.rotation.x = camera_orientation_robot_frame[0]
    transform_msg.transform.rotation.y = camera_orientation_robot_frame[1]
    transform_msg.transform.rotation.z = camera_orientation_robot_frame[2]
    transform_msg.transform.rotation.w = camera_orientation_robot_frame[3]
    self.broadcaster.sendTransform(transform_msg)


def main(argv):
  del argv  # unused
  rospy.init_node('tf_publisher', anonymous=True)

  publisher = TFPublisher()
  rospy.Subscriber("/robot_state", robot_state, publisher.robot_state_callback)
  rospy.Subscriber("/imu/data", Imu, publisher.camera_imu_callback)
  rospy.spin()


if __name__ == "__main__":
  app.run(main)
