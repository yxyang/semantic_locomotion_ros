#!/usr/bin/env python
"""Greedily follow GPS waypoints using PD control."""
from absl import app
from absl import flags

import cv2
from geographiclib.geodesic import Geodesic
from geometry_msgs.msg import PoseStamped
import matplotlib
import matplotlib.pyplot as plt
from nav_msgs.msg import Path
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage, NavSatFix
from std_msgs.msg import Bool, String
from tf.transformations import euler_from_quaternion
import tf2_py as tf2
import tf2_ros

flags.DEFINE_string('waypoint_file_path', None, 'path to waypoint file.')
FLAGS = flags.FLAGS


def compute_distance(lat1, lon1, lat2, lon2):
  trans = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)
  return trans['s12']


class PathGenerator:
  """Generates path from GPS waypoints."""
  def __init__(self, waypoints, checkpoint_reach_tolerance=4):
    self._waypoints = waypoints
    self._path_publisher = rospy.Publisher('/navigation/path',
                                           Path,
                                           queue_size=1)

    self._tf_buffer = tf2_ros.Buffer()
    self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)
    self._checkpoint_reach_tolerance = checkpoint_reach_tolerance
    self._waypoints_cartesian = None
    self._bev_map_publisher = rospy.Publisher('/navigation/map/compressed',
                                              CompressedImage,
                                              queue_size=1)
    self._distance_publisher = rospy.Publisher('/status/distance_to_waypoint',
                                               String,
                                               queue_size=1)

  def gps_fix_callback(self, fix):
    """Update path from GPS fix message."""
    if fix.status.status == fix.status.STATUS_NO_FIX:
      return
    # Remove checkpoints that have already been reached.
    while self._waypoints.shape[0] > 0:
      distance = compute_distance(self._waypoints[0, 0], self._waypoints[0, 1],
                                  fix.latitude, fix.longitude)
      if distance >= self._checkpoint_reach_tolerance:
        break
      self._waypoints = self._waypoints[1:]

    if self._waypoints.shape[0] == 0:
      return

    self._distance_publisher.publish(
        "Distance to next waypoint: {}".format(distance))

    # Get robot transform since we publish in robot frame
    try:
      trans = self._tf_buffer.lookup_transform('world', 'base_link',
                                               fix.header.stamp,
                                               rospy.Duration(1))
    except tf2.LookupException as ex:
      rospy.logwarn(ex)
      return
    except tf2.ExtrapolationException as ex:
      rospy.logwarn(ex)
      return
    trans = trans.transform
    robot_orientation_quaternion = [
        trans.rotation.x, trans.rotation.y, trans.rotation.z, trans.rotation.w
    ]
    angle_robot_world = euler_from_quaternion(robot_orientation_quaternion)[2]

    base_lat, base_lon = fix.latitude, fix.longitude

    path = Path()
    path.header.stamp = rospy.get_rostime()
    path.header.frame_id = 'base_link'
    waypoints, waypoints_cartesian = [], []
    for gps_waypoint in self._waypoints:
      lat, lon = gps_waypoint
      transform = Geodesic.WGS84.Inverse(base_lat, base_lon, lat, lon)
      distance = transform['s12']
      angle_gps_deg = transform['azi1']
      angle_gps_rad = angle_gps_deg / 180 * np.pi  # To radians
      # Original angle: clockwise from north
      # Need to convert to: counterclockwise from east
      angle_world_rad = -angle_gps_rad + np.pi / 2
      angle_robot_rad = angle_world_rad - angle_robot_world

      point_x = distance * np.cos(angle_robot_rad)
      point_y = distance * np.sin(angle_robot_rad)
      waypoint = PoseStamped()
      waypoint.header = path.header
      waypoint.pose.position.x = point_x
      waypoint.pose.position.y = point_y
      waypoint.pose.position.z = 0
      waypoint.pose.orientation.x = 0
      waypoint.pose.orientation.y = 0
      waypoint.pose.orientation.z = 0
      waypoint.pose.orientation.w = 1
      waypoints.append(waypoint)
      waypoints_cartesian.append((point_x, point_y))

    self._waypoints_cartesian = np.array(waypoints_cartesian)
    path.poses = waypoints
    self._path_publisher.publish(path)

  def generate_2d_visualization(self):
    """Generates BEV path map for visualization."""
    if self._waypoints_cartesian is None:
      return
    fig = plt.figure(figsize=(4, 4))
    plt.axis('off')
    plt.plot(self._waypoints_cartesian[:, 0], self._waypoints_cartesian[:, 1])
    plt.scatter(self._waypoints_cartesian[:1, 0],
                self._waypoints_cartesian[:1, 1],
                color='red')
    plt.scatter([0], [0])
    plt.axis('equal')
    fig.canvas.draw()
    image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] +
                                      (3,))
    msg = CompressedImage()
    msg.header.stamp = rospy.Time.now()
    msg.format = "png"
    msg.data = np.array(cv2.imencode(".png", image_array)[1]).tobytes()
    self._bev_map_publisher.publish(msg)
    plt.close(fig)

  def skip_waypoint_callback(self, msg):
    del msg # unused
    if len(self._waypoints) > 0:
      rospy.loginfo("Waypoint skipped.")
      self._waypoints = self._waypoints[1:]
    else:
      rospy.loginfo("No waypoint to skip.")


def main(argv):
  del argv  # unused
  matplotlib.use('Agg')
  plt.ioff()

  rospy.init_node('path_generator', anonymous=True)
  waypoints = np.load(open(FLAGS.waypoint_file_path, 'rb'))
  path_generator = PathGenerator(waypoints)
  rospy.Subscriber('/fix', NavSatFix, path_generator.gps_fix_callback)
  rospy.Subscriber('/skip_waypoint', Bool,
                   path_generator.skip_waypoint_callback)

  rate = rospy.Rate(1)
  while not rospy.is_shutdown():
    path_generator.generate_2d_visualization()
    rate.sleep()


if __name__ == "__main__":
  app.run(main)
