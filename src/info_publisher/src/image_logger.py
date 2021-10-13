#!/usr/bin/env python
"""Example of convex MPCcontroller on A1 robot."""
from datetime import datetime
import os

from absl import app
from absl import flags

import cv2
from cv_bridge import CvBridge
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage

flags.DEFINE_string('logdir', 'logs', 'logdir.')
flags.DEFINE_integer('max_num_images', 10000,
                     'Maximum number of images to record.')
flags.DEFINE_integer(
    'num_images_to_delete', 1000,
    'Number of images to delete when the buffer becomes full.')

FLAGS = flags.FLAGS


class ImageLogger:
  """Log segmentation results to file."""
  def __init__(self):
    self._bridge = CvBridge()
    self._camera_image = None

  def record_camera_image(self, image):
    self._camera_image = image

  def record_segmentation(self, segmentation):
    """Records segmentation and original image side by sdie."""
    if self._camera_image is None:
      return
    filename_postfix = datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')
    full_dir = os.path.join(
        FLAGS.logdir, 'log_{}_{}.png'.format(filename_postfix, 'segmentation'))

    np_arr = np.fromstring(segmentation.data, np.uint8)
    cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(full_dir, cv_image)

    full_dir = os.path.join(FLAGS.logdir,
                            'log_{}_{}.png'.format(filename_postfix, 'camera'))
    np_arr = np.fromstring(self._camera_image.data, np.uint8)
    cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(full_dir, cv_image)


def delete_old_files(logdir):
  files_list = list(os.listdir(logdir))
  if len(files_list) > FLAGS.max_num_images:
    rospy.loginfo("Image buffer limit reached, deleting...")
    files_list = sorted(files_list)
    for filename in files_list[:FLAGS.num_images_to_delete]:
      os.remove(os.path.join(logdir, filename))


def main(argv):
  del argv  # unused

  if not os.path.exists(FLAGS.logdir):
    os.makedirs(FLAGS.logdir)

  image_logger = ImageLogger()
  rospy.Subscriber("/perception/camera_image/compressed",
                   CompressedImage, image_logger.record_camera_image)
  rospy.Subscriber("/perception/segmentation_map/compressed",
                   CompressedImage, image_logger.record_segmentation)
  rospy.init_node("image_logger", anonymous=True)

  rate = rospy.Rate(0.1)
  while not rospy.is_shutdown():
    delete_old_files(FLAGS.logdir)
    rate.sleep()


if __name__ == "__main__":
  app.run(main)
