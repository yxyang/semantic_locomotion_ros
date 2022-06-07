#!/usr/bin/env python
"""Simulates a pointmass robot for demonstration purpose."""
from absl import app
from absl import flags

from geographiclib.geodesic import Geodesic
import geometry_msgs.msg
import numpy as np
import rospy
from sensor_msgs.msg import NavSatFix, NavSatStatus
from std_msgs.msg import String
import tf2_ros
import pybullet as p

from a1_interface.msg import robot_state

flags.DEFINE_string('gps_anchor_file', None, 'path to gps anchor coordinates.')
flags.DEFINE_bool('use_emulated_gps', True, 'whether to use emulated gps.')
FLAGS = flags.FLAGS


class RobotLocalizer:
  """A simple pointmass robot."""
  def __init__(self):
    self._px, self._py, self._heading, self._heading_bias = 0., 0., 0., 0.
    self._odometry_displacement = np.zeros(2)
    self._last_robot_state = None
    self._last_gps_fix = None

    self._tf_broadcaster = tf2_ros.TransformBroadcaster()
    self._gps_publisher = rospy.Publisher('/fix', NavSatFix, queue_size=1)
    self._gps_anchor_world_frame = None
    self._gps_anchor_lat_lon = None

    self._heading_bias_publisher = rospy.Publisher('/status/heading_bias',
                                                   String,
                                                   queue_size=1)

  def update_gps_anchor(self, latitude, longitude):
    self._gps_anchor_world_frame = [self._px, self._py]
    self._gps_anchor_lat_lon = [latitude, longitude]

  def emulated_robot_state_callback(self, robot_state_msg):
    """Broadcast TF transforms."""
    self._px = robot_state_msg.base_position[0]
    self._py = robot_state_msg.base_position[1]
    self._broadcast_tf()
    s12 = np.sqrt((self._px - self._gps_anchor_world_frame[0])**2 +
                  (self._py - self._gps_anchor_world_frame[1])**2)
    angle_world_rad = np.arctan2(self._py - self._gps_anchor_world_frame[1],
                                 self._px - self._gps_anchor_world_frame[0])
    # World frame: ccw from east
    # Gps frame: cw from north
    angle_gps_rad = np.mod(-angle_world_rad + np.pi / 2, 2 * np.pi)
    azi1 = angle_gps_rad / np.pi * 180

    new_coord = Geodesic.WGS84.Direct(lat1=self._gps_anchor_lat_lon[0],
                                      lon1=self._gps_anchor_lat_lon[1],
                                      azi1=azi1,
                                      s12=s12)
    fix = NavSatFix()
    fix.header.stamp = rospy.get_rostime()
    fix.header.frame_id = 'base_footprint'
    fix.status.status = NavSatStatus.STATUS_FIX
    fix.status.service = NavSatStatus.SERVICE_GPS
    fix.latitude = new_coord['lat2']
    fix.longitude = new_coord['lon2']
    self._gps_publisher.publish(fix)

  def real_robot_state_callback(self, robot_state_msg):
    """Computes odometry displacement from robot state."""
    if self._last_robot_state is not None:
      base_vels_body_frame = robot_state_msg.base_velocity
      base_yaw = p.getEulerFromQuaternion(
          robot_state_msg.base_orientation_quat_xyzw)[2]
      base_vx = base_vels_body_frame[0] * np.cos(
          base_yaw) - base_vels_body_frame[1] * np.sin(base_yaw)
      base_vy = base_vels_body_frame[0] * np.sin(
          base_yaw) + base_vels_body_frame[1] * np.cos(base_yaw)
      dt = robot_state_msg.timestamp - self._last_robot_state.timestamp
      self._odometry_displacement += np.array([base_vx, base_vy]) * np.clip(
          dt.to_sec(), 0, 0.1)

    self._last_robot_state = robot_state_msg

  def _broadcast_tf(self):
    """Broadcast TF message."""
    transform_msg = geometry_msgs.msg.TransformStamped()
    transform_msg.header.stamp = rospy.get_rostime()
    transform_msg.header.frame_id = "world"
    transform_msg.child_frame_id = "base_footprint"
    transform_msg.transform.translation.x = self._px
    transform_msg.transform.translation.y = self._py
    transform_msg.transform.translation.z = 0.
    transform_msg.transform.rotation.x = 0.
    transform_msg.transform.rotation.y = 0.
    transform_msg.transform.rotation.z = np.sin(self._heading / 2)
    transform_msg.transform.rotation.w = np.cos(self._heading / 2)
    self._tf_broadcaster.sendTransform(transform_msg)

  def gps_fix_callback(self, fix):
    """Update path from GPS fix message."""
    if fix.status.status == fix.status.STATUS_NO_FIX:
      return

    base_lat, base_lon = fix.latitude, fix.longitude
    transform = Geodesic.WGS84.Inverse(self._gps_anchor_lat_lon[0],
                                       self._gps_anchor_lat_lon[1], base_lat,
                                       base_lon)
    distance = transform['s12']
    angle_gps_deg = transform['azi1']
    angle_gps_rad = angle_gps_deg / 180 * np.pi  # To radians
    # Original angle: clockwise from north
    # Need to convert to: counterclockwise from east
    angle_world_rad = -angle_gps_rad + np.pi / 2
    self._px = distance * np.cos(angle_world_rad)
    self._py = distance * np.sin(angle_world_rad)

    # Update heading bias
    if self._last_gps_fix is not None:
      transform = Geodesic.WGS84.Inverse(self._last_gps_fix.latitude,
                                         self._last_gps_fix.longitude,
                                         base_lat, base_lon)
      distance = transform['s12']
      angle_gps_deg = transform['azi1']
      angle_gps_rad = angle_gps_deg / 180 * np.pi  # To radians
      # Original angle: clockwise from north
      # Need to convert to: counterclockwise from east
      angle_world_rad = -angle_gps_rad + np.pi / 2

      angle_robot_rad = np.arctan2(self._odometry_displacement[1],
                                   self._odometry_displacement[0])
      self._odometry_displacement = np.zeros(2)
      new_heading_bias = angle_world_rad - angle_robot_rad
      if distance > 0.2:  # Only consider valid displacements.
        self._heading_bias = 0.95 * self._heading_bias + 0.05 * new_heading_bias

    self._heading_bias_publisher.publish("IMU yaw bias: {:.2f}".format(
        self._heading_bias))

    if self._last_robot_state is not None:
      robot_heading = p.getEulerFromQuaternion(
          self._last_robot_state.base_orientation_quat_xyzw)[2]
      self._heading = np.mod(robot_heading + self._heading_bias, 2 * np.pi)
    self._broadcast_tf()

    self._last_gps_fix = fix


def main(argv):
  del argv  # unused
  rospy.init_node("pointmass_robot", anonymous=True)

  waypoints_file = np.load(open(FLAGS.gps_anchor_file, 'rb'))
  localizer = RobotLocalizer()
  localizer.update_gps_anchor(waypoints_file[0, 0], waypoints_file[0, 1])
  if FLAGS.use_emulated_gps:
    rospy.Subscriber("/robot_state", robot_state,
                     localizer.emulated_robot_state_callback)
  else:
    rospy.Subscriber("/robot_state", robot_state,
                     localizer.real_robot_state_callback)
    rospy.Subscriber('/fix', NavSatFix, localizer.gps_fix_callback)

  rospy.spin()


if __name__ == "__main__":
  app.run(main)
