#!/usr/bin/env python
"""Example of convex MPCcontroller on A1 robot."""
from absl import app
from absl import flags

import cv2
from cv_bridge import CvBridge
from datetime import datetime
import os
import rospy
from sensor_msgs.msg import Image

flags.DEFINE_string('logdir', 'logs', 'logdir.')

FLAGS = flags.FLAGS


class ImageLogger:
  """Log segmentation results to file."""
  def __init__(self):
    self._bridge = CvBridge()

  def record_segmentation(self, segmentation):
    filename_postfix = datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')
    full_dir = os.path.join(
        FLAGS.logdir, '{}_{}.png'.format('segmentation', filename_postfix))
    cv_image = self._bridge.imgmsg_to_cv2(segmentation,
                                          desired_encoding='passthrough')
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(full_dir, cv_image)


def main(argv):
  del argv  # unused

  if not os.path.exists(FLAGS.logdir):
    os.makedirs(FLAGS.logdir)

  image_logger = ImageLogger()
  rospy.Subscriber("/perception/segmentation_map", Image,
                   image_logger.record_segmentation)
  rospy.init_node("image_logger", anonymous=True)
  rospy.spin()


if __name__ == "__main__":
  app.run(main)
