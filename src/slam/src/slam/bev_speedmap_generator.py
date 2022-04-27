#!/usr/bin/env python
"""Generate BEV speed map from pointcloud speedmap."""
from absl import app
from absl import flags

import numpy as np
import ros_numpy
import rospy
from sensor_msgs.msg import PointCloud2
import tf2_py as tf2
import tf2_ros
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

FLAGS = flags.FLAGS


class BEVSpeedMapGenerator:
  """Generates BEV speed map from pointcloud speedmap."""
  def __init__(
      self,
      ground_frame_id='base_footprint',
      height_tolerance=0.05,
      map_width=2,  # Unit: meters
      map_length=2,
      resolution=0.1):
    self._height_tolerance = height_tolerance
    self._ground_frame_id = ground_frame_id
    self._map_width = map_width
    self._map_length = map_length
    self._resolution = resolution
    self._num_bins_x = int(self._map_length / self._resolution)
    self._num_bins_y = int(self._map_width / self._resolution)

    self._tf_buffer = tf2_ros.Buffer()
    self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)
    self._occupancy_grid_publisher = rospy.Publisher(
        "/perception/speedmap/occupancygrid", PointCloud2, queue_size=1)

  def speedmap_pointcloud_callback(self, msg):
    """Converts from BEV speedcloud to BEV speedmap."""
    try:
      trans = self._tf_buffer.lookup_transform(self._ground_frame_id,
                                               msg.header.frame_id,
                                               msg.header.stamp,
                                               rospy.Duration(1))
    except tf2.LookupException as ex:
      rospy.logwarn(ex)
      return
    except tf2.ExtrapolationException as ex:
      rospy.logwarn(ex)
      return

    cloud_ground_frame = do_transform_cloud(msg, trans)
    points_array = ros_numpy.point_cloud2.pointcloud2_to_array(
        cloud_ground_frame)
    condition = ((points_array['x'] >= 0) &
                 (points_array['x'] <= self._map_length) &
                 (points_array['y'] >= -self._map_width / 2) &
                 (points_array['y'] <= self._map_width / 2) &
                 (points_array['z'] >= -self._height_tolerance) &
                 (points_array['z'] <= self._height_tolerance))
    useful_points = points_array[np.where(condition)]
    speed_sum, x_edges, y_edges = np.histogram2d(
        useful_points['x'],
        useful_points['y'],
        bins=[self._num_bins_x, self._num_bins_y],
        range=[[0, self._map_length],
               [-self._map_width / 2, self._map_width / 2]],
        weights=useful_points['speed'])
    speed_count, _, _ = np.histogram2d(
        useful_points['x'],
        useful_points['y'],
        bins=[self._num_bins_x, self._num_bins_y],
        range=[[0, self._map_length],
               [-self._map_width / 2, self._map_width / 2]])
    avg_speed = (speed_sum / (speed_count + 1e-7)).T

    grid_coord_x, grid_coord_y = np.meshgrid(np.arange(self._num_bins_x),
                                             np.arange(self._num_bins_y),
                                             indexing='xy')
    pos_x, pos_y = x_edges[grid_coord_x], y_edges[grid_coord_y]
    pos_z = np.zeros_like(pos_x)
    point_cloud_array = np.stack((pos_x, pos_y, pos_z, avg_speed),
                                 axis=-1).astype(np.float32)
    dtype = np.dtype([('x', point_cloud_array.dtype),
                      ('y', point_cloud_array.dtype),
                      ('z', point_cloud_array.dtype),
                      ('speed', point_cloud_array.dtype)])
    cloud_msg = PointCloud2()
    cloud_msg.header.stamp = msg.header.stamp
    cloud_msg.header.frame_id = self._ground_frame_id
    cloud_msg.height = point_cloud_array.shape[0]
    cloud_msg.width = point_cloud_array.shape[1]
    cloud_msg.fields = ros_numpy.point_cloud2.dtype_to_fields(dtype)
    cloud_msg.is_bigendian = False  # assumption
    cloud_msg.point_step = dtype.itemsize
    cloud_msg.row_step = cloud_msg.point_step * point_cloud_array.shape[1]
    cloud_msg.is_dense = np.isfinite(point_cloud_array).all()
    cloud_msg.data = point_cloud_array.tobytes()
    self._occupancy_grid_publisher.publish(cloud_msg)


def main(argv):
  del argv  # unused
  rospy.init_node('bev_speedmap_generator', anonymous=True)
  speedmap_generator = BEVSpeedMapGenerator(height_tolerance=1,
                                            resolution=0.05)
  rospy.Subscriber('/perception/speedmap/pointcloud', PointCloud2,
                   speedmap_generator.speedmap_pointcloud_callback)
  rospy.spin()


if __name__ == "__main__":
  app.run(main)
