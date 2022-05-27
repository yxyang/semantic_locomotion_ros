#!/usr/bin/env python
"""Example for real-time semantic segmentation using realsense camera."""
from absl import app
from absl import flags

import numpy as np
import pyrealsense2 as rs
import rospy

import ros_numpy
from sensor_msgs.msg import Image

flags.DEFINE_integer('frame_width', 424, 'frame width.')
flags.DEFINE_integer('frame_height', 240, 'frame height.')
flags.DEFINE_integer('frame_rate', 6, 'frame rate.')
FLAGS = flags.FLAGS


def main(_):
  pipeline = rs.pipeline()
  config = rs.config()
  config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 60)
  pipeline.start(config)

  camera_image_publisher = rospy.Publisher(
      '/camera/color/image_raw', Image, queue_size=1)
  rospy.init_node('realsense_camera_capture', anonymous=True)
  rate = rospy.Rate(FLAGS.frame_rate)
  while not rospy.is_shutdown():
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.array(color_frame.get_data())
    msg = ros_numpy.image.numpy_to_image(color_image, 'bgr8')
    camera_image_publisher.publish(msg)
    rate.sleep()
  pipeline.stop()


if __name__ == "__main__":
  app.run(main)
