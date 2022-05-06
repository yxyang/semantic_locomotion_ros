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

from a1_interface.msg import speed_command

flags.DEFINE_float('control_timestep', 0.02, 'the control timestep.')
flags.DEFINE_string('gps_anchor_file', None, 'path to gps anchor coordinates.')
FLAGS = flags.FLAGS


class PointmassRobot:
  """A simple pointmass robot."""
  def __init__(self, dt=0.002, use_emulated_gps=True):
    self._px, self._py, self._heading = 0., 0., 0.
    self._vx, self._vy, self._rot_z = 0., 0., 0.
    self._dt = dt
    self._tf_broadcaster = tf2_ros.TransformBroadcaster()
    self._use_emulated_gps = use_emulated_gps
    self._gps_publisher = rospy.Publisher('/fix', NavSatFix, queue_size=1)
    self._gps_anchor_world_frame = None
    self._gps_anchor_lat_long = None

  def receive_speed_command(self, command):
    self._vx = command.vel_x
    self._vy = command.vel_y
    self._rot_z = command.rot_z

  def update_gps_anchor(self, latitude, longitude):
    self._gps_anchor_world_frame = [self._px, self._py]
    self._gps_anchor_lat_long = [latitude, longitude]

  def step_simulation(self):
    """Simulate the pointmass robot from joystick commands."""
    vx_world_frame = self._vx * np.cos(self._heading) - self._vy * np.sin(
        self._heading)
    vy_world_frame = self._vx * np.sin(self._heading) + self._vy * np.cos(
        self._heading)
    self._px += vx_world_frame * self._dt
    self._py += vy_world_frame * self._dt
    self._heading += self._rot_z * self._dt
    self._broadcast_tf()
    if self._use_emulated_gps:
      self._broadcast_gps()

  def _broadcast_tf(self):
    """Broadcast TF transforms."""
    transform_msg = geometry_msgs.msg.TransformStamped()
    transform_msg.header.stamp = rospy.get_rostime()
    transform_msg.header.frame_id = "world"
    transform_msg.child_frame_id = "base_link"
    transform_msg.transform.translation.x = self._px
    transform_msg.transform.translation.y = self._py
    transform_msg.transform.translation.z = 0.
    transform_msg.transform.rotation.x = 0.
    transform_msg.transform.rotation.y = 0.
    transform_msg.transform.rotation.z = np.sin(self._heading / 2)
    transform_msg.transform.rotation.w = np.cos(self._heading / 2)
    self._tf_broadcaster.sendTransform(transform_msg)

  def _broadcast_gps(self):
    """Broadcast emulated GPS coordinate."""
    s12 = np.sqrt((self._px - self._gps_anchor_world_frame[0])**2 +
                  (self._py - self._gps_anchor_world_frame[1])**2)
    angle_world_rad = np.arctan2(self._py - self._gps_anchor_world_frame[1],
                                 self._px - self._gps_anchor_world_frame[0])
    # World frame: ccw from east
    # Gps frame: cw from north
    angle_gps_rad = np.mod(-angle_world_rad + np.pi / 2, 2 * np.pi)
    azi1 = angle_gps_rad / np.pi * 180

    new_coord = Geodesic.WGS84.Direct(lat1=self._gps_anchor_lat_long[0],
                                      lon1=self._gps_anchor_lat_long[1],
                                      azi1=azi1,
                                      s12=s12)
    fix = NavSatFix()
    fix.header.stamp = rospy.get_rostime()
    fix.header.frame_id = 'base_link'
    fix.status.status = NavSatStatus.STATUS_FIX
    fix.status.service = NavSatStatus.SERVICE_GPS
    fix.latitude = new_coord['lat2']
    fix.longitude = new_coord['lon2']
    self._gps_publisher.publish(fix)


def main(argv):
  del argv  # unused
  rospy.init_node("pointmass_robot", anonymous=True)

  if FLAGS.gps_anchor_file is not None:
    waypoints_file = np.load(open(FLAGS.gps_anchor_file, 'rb'))
    robot = PointmassRobot(FLAGS.control_timestep, use_emulated_gps=True)
    robot.update_gps_anchor(waypoints_file[0, 0], waypoints_file[0, 1])
  else:
    robot = PointmassRobot(FLAGS.control_timestep, use_emulated_gps=False)

  rospy.Subscriber("speed_command", speed_command, robot.receive_speed_command)

  rate = rospy.Rate(1 / FLAGS.control_timestep)
  while not rospy.is_shutdown():
    robot.step_simulation()
    rate.sleep()


if __name__ == "__main__":
  app.run(main)
