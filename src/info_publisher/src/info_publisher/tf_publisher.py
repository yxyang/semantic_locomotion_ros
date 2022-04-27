#!/usr/bin/env python
"""Subscribes to robot state and publish as TF transforms."""

from absl import app
from absl import flags

from cv_bridge import CvBridge
import geometry_msgs.msg
from image_geometry import PinholeCameraModel
import numpy as np
from sensor_msgs.msg import CameraInfo, Image, Imu, PointCloud2
import ros_numpy
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
    self._camera_model = PinholeCameraModel()
    self._camera_model.initialized = False
    self._speed_map_array = None
    self._cv_bridge = CvBridge()
    self._pointcloud_publisher = rospy.Publisher(
        "/perception/speedmap/pointcloud", PointCloud2, queue_size=1)

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

  def depth_info_callback(self, camera_info):
    self._camera_model.fromCameraInfo(camera_info)
    self._camera_model.initialized = True

  def depth_image_callback(self, image):
    """Publish pointcloud with speed estimation."""
    if not self._camera_model.initialized:
      return
    projection_matrix = self._camera_model.projectionMatrix()
    fx, fy = projection_matrix[0, 0], projection_matrix[1, 1]
    cx, cy = projection_matrix[0, 2], projection_matrix[1, 2]
    tx, ty = projection_matrix[0, 3], projection_matrix[1, 3]

    depth_image = self._cv_bridge.imgmsg_to_cv2(image,
                                                desired_encoding="passthrough")
    depth_array_mm = np.array(depth_image, dtype=np.float32)  # Depth in mm
    depth_array_m = depth_array_mm / 1000.
    height, width = image.height, image.width
    image_coord_y, image_coord_x = np.meshgrid(np.arange(height),
                                               np.arange(width),
                                               indexing='ij')
    ray_x = (image_coord_x - cx - tx) / fx
    ray_y = (image_coord_y - cy - ty) / fy
    camera_coord_x = ray_x * depth_array_m
    camera_coord_y = ray_y * depth_array_m
    camera_coord_z = depth_array_m
    if self._speed_map_array is not None:
      point_cloud_array = np.stack((camera_coord_x, camera_coord_y,
                                    camera_coord_z, self._speed_map_array),
                                   axis=-1).astype(np.float32)
      dtype = np.dtype([('x', point_cloud_array.dtype),
                        ('y', point_cloud_array.dtype),
                        ('z', point_cloud_array.dtype),
                        ('speed', point_cloud_array.dtype)])
    else:
      point_cloud_array = np.stack(
          (camera_coord_x, camera_coord_y, camera_coord_z),
          axis=-1).astype(np.float32)
      dtype = np.dtype([('x', point_cloud_array.dtype),
                        ('y', point_cloud_array.dtype),
                        ('z', point_cloud_array.dtype)])
    cloud_msg = PointCloud2()
    cloud_msg.header.stamp = image.header.stamp
    cloud_msg.header.frame_id = image.header.frame_id
    cloud_msg.height = point_cloud_array.shape[0]
    cloud_msg.width = point_cloud_array.shape[1]

    # Note that x and y here are interchanged to account for changes in
    # coordinate conventions.

    cloud_msg.fields = ros_numpy.point_cloud2.dtype_to_fields(dtype)
    cloud_msg.is_bigendian = False  # assumption
    cloud_msg.point_step = dtype.itemsize
    cloud_msg.row_step = cloud_msg.point_step * point_cloud_array.shape[1]
    cloud_msg.is_dense = np.isfinite(point_cloud_array).all()
    cloud_msg.data = point_cloud_array.tobytes()
    self._pointcloud_publisher.publish(cloud_msg)

  def speed_map_callback(self, image):
    self._speed_map_array = ros_numpy.image.image_to_numpy(image)


def main(argv):
  del argv  # unused
  rospy.init_node('tf_publisher', anonymous=True)

  publisher = TFPublisher()
  rospy.Subscriber("/robot_state", robot_state, publisher.robot_state_callback)
  rospy.Subscriber("/imu/data", Imu, publisher.camera_imu_callback)
  rospy.Subscriber("/camera/depth/camera_info", CameraInfo,
                   publisher.depth_info_callback)
  rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image,
                   publisher.depth_image_callback)
  rospy.Subscriber("/perception/speedmap/image_raw", Image,
                   publisher.speed_map_callback)
  rospy.spin()


if __name__ == "__main__":
  app.run(main)
