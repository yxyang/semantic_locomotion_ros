#!/usr/bin/env python
"""Greedily follow GPS waypoints using PD control."""
from absl import app
from absl import flags

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import numpy as np
from geographiclib.geodesic import Geodesic
import rospy
from sensor_msgs.msg import NavSatFix
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
  def __init__(self, waypoints, checkpoint_reach_tolerance=1):
    self._waypoints = waypoints
    self._path_publisher = rospy.Publisher('/path', Path, queue_size=1)

    self._tf_buffer = tf2_ros.Buffer()
    self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)
    self._checkpoint_reach_tolerance = checkpoint_reach_tolerance

  def gps_fix_callback(self, fix):
    """Update path from GPS fix message."""
    # Remove checkpoints that have already been reached.
    while (self._waypoints.shape[0] > 0) and (compute_distance(
        self._waypoints[0, 0], self._waypoints[0, 1], fix.latitude,
        fix.longitude) < self._checkpoint_reach_tolerance):
      self._waypoints = self._waypoints[1:]

    if self._waypoints.shape[0] == 0:
      return

    # Get robot transform since we publish in robot frame
    try:
      trans = self._tf_buffer.lookup_transform('world', fix.header.frame_id,
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
    waypoints = []
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

    path.poses = waypoints
    self._path_publisher.publish(path)


def main(argv):
  del argv  # unused
  rospy.init_node('path_generator', anonymous=True)
  waypoints = np.load(open(FLAGS.waypoint_file_path, 'rb'))
  path_generator = PathGenerator(waypoints)
  rospy.Subscriber('/fix', NavSatFix, path_generator.gps_fix_callback)
  rospy.spin()


if __name__ == "__main__":
  app.run(main)
