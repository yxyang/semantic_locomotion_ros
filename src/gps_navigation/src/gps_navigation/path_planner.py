#!/usr/bin/env python
"""A simple path planner for left/mid/right."""
from absl import app
from absl import flags

import cv2
from nav_msgs.msg import Path
import numpy as np
import ros_numpy
import rospy
from sensor_msgs.msg import Image

from a1_interface.msg import speed_command

FLAGS = flags.FLAGS

ROTATION_MAP = [0.5, 0, -0.5]  # Left, Straight, Right


def compute_end_point(angle, x_lim=40, y_lim=20):
  slope = np.tan(angle)
  multiplier = np.sign(np.cos(angle))
  if slope < -1:
    return np.array([-y_lim / slope, -y_lim]) * multiplier
  elif slope > 1:
    return np.array([y_lim / slope, y_lim]) * multiplier
  else:
    return np.array([x_lim, slope * x_lim]) * multiplier


def generate_masks():
  """Generate masks for regions of interest for left/straight/right."""
  origin = np.array([0, 20])
  angles = np.array([29, 10, -10, -29]) / 180 * np.pi
  end_points = np.array([compute_end_point(angle)
                         for angle in angles]) + origin
  masks = []
  for idx in range(len(end_points) - 1):
    mask = np.zeros((40, 40)).astype(np.int32)
    cv2.fillConvexPoly(
        mask,
        np.array([origin, end_points[idx], end_points[idx + 1],
                  origin]).astype(np.int32), 1)
    masks.append(mask)
  return masks


class PathPlanner:
  """Simple path planner with left/straight/right actions."""
  def __init__(self,
               local_distance=2.,
               action_angles=np.array([30., 0., -30.]) / 180 * np.pi,
               remaining_speed=0.8):
    self._local_distance = local_distance
    self._action_angles = action_angles
    self._local_endpoints = np.stack([
        self._local_distance * np.cos(self._action_angles),
        self._local_distance * np.sin(self._action_angles)
    ],
                                     axis=1)
    self._remaining_speed = remaining_speed
    self._masks = generate_masks()
    self._goal = np.array([0., 0.])

    self._speed_command_publisher = rospy.Publisher(
        "/navigation/speed_command", speed_command, queue_size=1)

  def path_callback(self, path_msg):
    goal_position = path_msg.poses[0].pose.position
    self._goal = np.array([goal_position.x, goal_position.y])

  def speedmap_callback(self, speedmap_msg):
    """Generates speed command from perceived speedmap."""
    speedmap = ros_numpy.image.image_to_numpy(speedmap_msg)
    local_speeds = np.array(
        [np.sum(mask * speedmap) / np.sum(mask) for mask in self._masks])
    local_speeds = np.clip(local_speeds, 1e-7, 2)
    local_traversal_time = self._local_distance / local_speeds

    goal_displacement = self._goal - self._local_endpoints
    goal_distance = np.sqrt(np.sum(np.square(goal_displacement), axis=1))
    remaining_traversal_time = goal_distance / self._remaining_speed
    total_traversal_time = local_traversal_time + remaining_traversal_time
    action = np.argmin(total_traversal_time)

    speed_command_msg = speed_command()
    speed_command_msg.timestamp = rospy.get_rostime()
    speed_command_msg.vel_x = local_speeds[action]
    speed_command_msg.vel_y = 0.
    speed_command_msg.rot_z = ROTATION_MAP[action]
    self._speed_command_publisher.publish(speed_command_msg)


def main(argv):
  del argv  # unused
  rospy.init_node("path_planner", anonymous=True)

  planner = PathPlanner()
  rospy.Subscriber('/navigation/path', Path, planner.path_callback)
  rospy.Subscriber('/perception/bev_speedmap/image_raw', Image,
                   planner.speedmap_callback)

  rospy.spin()


if __name__ == "__main__":
  app.run(main)
