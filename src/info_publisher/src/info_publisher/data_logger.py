#!/usr/bin/env python
"""Example of convex MPCcontroller on A1 robot."""
from datetime import datetime
import getpass
import os
import pickle
import time

from absl import app
from absl import flags

import cv2
from cv_bridge import CvBridge
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from a1_interface.msg import controller_mode, robot_state

flags.DEFINE_string('logdir', 'ghost_memory', 'logdir.')
flags.DEFINE_integer('max_num_images', 100000,
                     'Maximum number of images to record.')
flags.DEFINE_integer(
    'num_images_to_delete', 1000,
    'Number of images to delete when the buffer becomes full.')

FLAGS = flags.FLAGS


class DataLogger:
  """Log segmentation results to file."""
  def __init__(self, logdir):
    self._logdir = logdir
    self._camera_image = None
    self._robot_state = None
    self._depth_image = None
    self._log_state_publisher = rospy.Publisher('status/data_logger',
                                                String,
                                                queue_size=1)
    self._cv_bridge = CvBridge()

  def record_robot_state(self, robot_state_data):
    self._robot_state = robot_state_data

  def record_camera_image(self, image):
    """Records camera image and saves related data."""
    if self._robot_state is None:
      self._log_state_publisher.publish("No Robot State")
      return

    if self._robot_state.controller_mode != controller_mode.WALK:
      self._log_state_publisher.publish("Robot not walking")
      return

    curr_time = datetime.now()
    folder_name = curr_time.strftime('%Y_%m_%d_%H_%M')
    filename_postfix = curr_time.strftime("%S_%f")

    full_folder = os.path.join(self._logdir, folder_name)
    if not os.path.exists(full_folder):
      os.makedirs(full_folder)

    full_dir = os.path.join(full_folder,
                            'log_{}_{}.png'.format(filename_postfix, 'camera'))
    np_arr = np.fromstring(image.data, np.uint8)
    self._camera_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    cv2.imwrite(full_dir, self._camera_image)

    full_dir = os.path.join(
        full_folder, 'log_{}_{}.pkl'.format(filename_postfix, 'robot_state'))
    pickle.dump(self._robot_state, open(full_dir, 'wb'))
    self._log_state_publisher.publish("Last frame: {}".format(
        datetime.now().strftime('%H:%M:%S')))


def delete_old_files(logdir):
  files_list = list(os.listdir(logdir))
  if len(files_list) > FLAGS.max_num_images:
    rospy.loginfo("Image buffer limit reached, deleting...")
    files_list = sorted(files_list)
    for filename in files_list[:FLAGS.num_images_to_delete]:
      os.remove(os.path.join(logdir, filename))


def main(argv):
  del argv  # unused
  rospy.init_node("data_logger", anonymous=True)

  logdir = os.path.join("/home", getpass.getuser(), FLAGS.logdir)

  if not os.path.exists(logdir):
    os.makedirs(logdir)

  data_logger = DataLogger(logdir)
  rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage,
                   data_logger.record_camera_image)
  rospy.Subscriber("/robot_state", robot_state, data_logger.record_robot_state)

  while not rospy.is_shutdown():
    delete_old_files(logdir)
    time.sleep(10)


if __name__ == "__main__":
  app.run(main)
