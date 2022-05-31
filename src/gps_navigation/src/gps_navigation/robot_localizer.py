#!/usr/bin/env python
"""Simulates a pointmass robot for demonstration purpose."""
from absl import app
from absl import flags

from geographiclib.geodesic import Geodesic
import geometry_msgs.msg
import numpy as np
import rospy
from sensor_msgs.msg import NavSatFix, NavSatStatus
import tf2_ros

from a1_interface.msg import robot_state

flags.DEFINE_string('gps_anchor_file', None, 'path to gps anchor coordinates.')
flags.DEFINE_bool('use_emulated_gps', True, 'whether to use emulated gps.')
FLAGS = flags.FLAGS


class RobotLocalizer:
  """A simple pointmass robot."""
  def __init__(self, dt=0.002):
    self._px, self._py, self._heading = 0., 0., 0.
    self._vx, self._vy, self._rot_z = 0., 0., 0.
    self._dt = dt
    self._tf_broadcaster = tf2_ros.TransformBroadcaster()
    self._gps_publisher = rospy.Publisher('/fix', NavSatFix, queue_size=1)
    self._gps_anchor_world_frame = None
    self._gps_anchor_lat_lon = None

  def speed_command_callback(self, command):
    self._vx = command.vel_x
    self._vy = command.vel_y
    self._rot_z = command.rot_z

  def update_gps_anchor(self, latitude, longitude):
    self._gps_anchor_world_frame = [self._px, self._py]
    self._gps_anchor_lat_lon = [latitude, longitude]

  def robot_state_callback(self, robot_state_msg):
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
    fix.header.frame_id = 'base_flat'
    fix.status.status = NavSatStatus.STATUS_FIX
    fix.status.service = NavSatStatus.SERVICE_GPS
    fix.latitude = new_coord['lat2']
    fix.longitude = new_coord['lon2']
    self._gps_publisher.publish(fix)

  def _broadcast_tf(self):
    """Broadcast TF message."""
    transform_msg = geometry_msgs.msg.TransformStamped()
    transform_msg.header.stamp = rospy.get_rostime()
    transform_msg.header.frame_id = "world"
    transform_msg.child_frame_id = "base_flat"
    transform_msg.transform.translation.x = self._px
    transform_msg.transform.translation.y = self._py
    transform_msg.transform.translation.z = 0.
    transform_msg.transform.rotation.x = 0.
    transform_msg.transform.rotation.y = 0.
    transform_msg.transform.rotation.z = 0.
    transform_msg.transform.rotation.w = 1.
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
    self._broadcast_tf()


def main(argv):
  del argv  # unused
  rospy.init_node("pointmass_robot", anonymous=True)

  waypoints_file = np.load(open(FLAGS.gps_anchor_file, 'rb'))
  localizer = RobotLocalizer()
  localizer.update_gps_anchor(waypoints_file[0, 0], waypoints_file[0, 1])
  if FLAGS.use_emulated_gps:
    rospy.Subscriber("/robot_state", robot_state,
                     localizer.robot_state_callback)
  else:
    rospy.Subscriber('/fix', NavSatFix, localizer.gps_fix_callback)

  rospy.spin()


if __name__ == "__main__":
  app.run(main)
