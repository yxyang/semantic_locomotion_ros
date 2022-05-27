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
  rospy.init_node('realsense_camera_capture', anonymous=True)

  pipeline = rs.pipeline()
  config = rs.config()
  config.enable_stream(rs.stream.color, FLAGS.frame_width, FLAGS.frame_height,
                       rs.format.bgr8, 60)
  config.enable_stream(rs.stream.depth, FLAGS.frame_width, FLAGS.frame_height,
                       rs.format.z16, 60)
  pipeline.start(config)
  # Align depth to color
  align = rs.align(rs.stream.color)

  color_image_publisher = rospy.Publisher('/camera/color/image_raw',
                                          Image,
                                          queue_size=1)
  depth_image_publisher = rospy.Publisher(
      '/camera/aligned_depth_to_color/image_raw', Image, queue_size=1)

  rate = rospy.Rate(FLAGS.frame_rate)
  while not rospy.is_shutdown():
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not aligned_depth_frame or not color_frame:
      continue

    color_image = np.array(color_frame.get_data())
    msg = ros_numpy.image.numpy_to_image(color_image, 'bgr8')
    color_image_publisher.publish(msg)

    depth_image = np.array(aligned_depth_frame.get_data())
    msg = ros_numpy.image.numpy_to_image(depth_image, 'mono16')
    depth_image_publisher.publish(msg)
    rate.sleep()
  pipeline.stop()


if __name__ == "__main__":
  app.run(main)
