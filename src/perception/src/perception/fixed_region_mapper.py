#!/usr/bin/env python
"""Computes traversability score by averaging semantics over fixed region."""
import cv2
import numpy as np
import rospy
from absl import app
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32

from perception.fixed_region_mapper_lib import FixedRegionMapper


def image_callback(image):
  im = np.frombuffer(image.data,
                     dtype=np.uint8).reshape(image.height, image.width,
                                             -1).astype(np.float32)
  im /= 255
  cv2.imshow("image", cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
  cv2.waitKey(1)


def main(argv):
  del argv  # unused
  mapper = FixedRegionMapper()
  rospy.Subscriber("/perception/camera_image/compressed", CompressedImage,
                   mapper.set_camera_image)

  segmentation_map_publisher = rospy.Publisher(
      '/perception/segmentation_map/compressed', CompressedImage, queue_size=1)
  traversability_score_publisher = rospy.Publisher(
      '/perception/traversability_score', Float32, queue_size=1)
  rospy.init_node("segmentation", anonymous=True)
  rate = rospy.Rate(6)
  while not rospy.is_shutdown():
    if mapper.image_array is not None:
      score, segmentation_map = mapper.get_segmentation_result()
      segmentation_map_publisher.publish(segmentation_map)
      traversability_score_publisher.publish(score)
      rospy.loginfo("Score: {}".format(score))
    rate.sleep()


if __name__ == "__main__":
  app.run(main)
