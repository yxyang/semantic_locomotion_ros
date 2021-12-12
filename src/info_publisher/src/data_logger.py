#!/usr/bin/env python
"""Example of convex MPCcontroller on A1 robot."""
from datetime import datetime
import getpass
import os
import pickle

from absl import app
from absl import flags

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
import time

from a1_interface.msg import robot_state

flags.DEFINE_string('usb_drive_name', 'ghost_memory', 'logdir.')
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

  def record_robot_state(self, robot_state_data):
    self._robot_state = robot_state_data

  def record_camera_image(self, image):
    self._camera_image = image

  def record_segmentation(self, segmentation):
    """Records segmentation and original image side by sdie."""
    if self._camera_image is None:
      rospy.loginfo("No camera image captured, skipping...")
      return

    if self._robot_state is None:
      rospy.loginfo("No robot state captured, skipping...")
      return

    filename_postfix = datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')
    full_dir = os.path.join(
        self._logdir, 'log_{}_{}.png'.format(filename_postfix, 'segmentation'))

    np_arr = np.fromstring(segmentation.data, np.uint8)
    cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(full_dir, cv_image)

    full_dir = os.path.join(self._logdir,
                            'log_{}_{}.png'.format(filename_postfix, 'camera'))
    np_arr = np.fromstring(self._camera_image.data, np.uint8)
    cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(full_dir, cv_image)

    full_dir = os.path.join(
        self._logdir, 'log_{}_{}.pkl'.format(filename_postfix, 'robot_state'))
    pickle.dump(self._robot_state, open(full_dir, 'wb'))


def delete_old_files(logdir):
  files_list = list(os.listdir(logdir))
  if len(files_list) > FLAGS.max_num_images:
    rospy.loginfo("Image buffer limit reached, deleting...")
    files_list = sorted(files_list)
    for filename in files_list[:FLAGS.num_images_to_delete]:
      os.remove(os.path.join(logdir, filename))


def main(argv):
  del argv  # unused

  logdir = os.path.join("/media", getpass.getuser(), FLAGS.usb_drive_name)

  if not os.path.exists(logdir):
    rospy.loginfo("Logdir: {}".format(logdir))
    raise RuntimeError("No USB drive with the name {} exists.".format(
        FLAGS.usb_drive_name))

  logdir = os.path.join(logdir, "data")

  if not os.path.exists(logdir):
    os.makedirs(logdir)
  data_logger = DataLogger(logdir)
  rospy.Subscriber("/perception/camera_image/compressed", CompressedImage,
                   data_logger.record_camera_image)
  rospy.Subscriber("/perception/segmentation_map/compressed", CompressedImage,
                   data_logger.record_segmentation)
  rospy.Subscriber("/robot_state", robot_state, data_logger.record_robot_state)
  rospy.init_node("data_logger", anonymous=True)

  while not rospy.is_shutdown():
    delete_old_files(logdir)
    time.sleep(10)


if __name__ == "__main__":
  app.run(main)
