#!/usr/bin/env python
"""Example for real-time semantic segmentation using realsense camera."""
from absl import app
from absl import flags

import cv2
import numpy as np
import pyrealsense2 as rs
import ros_numpy
import rospy
from sensor_msgs.msg import Image

flags.DEFINE_integer('frame_width', 640, 'frame width.')
flags.DEFINE_integer('frame_height', 360, 'frame height.')
flags.DEFINE_integer('frame_rate', 10, 'frame rate.')
FLAGS = flags.FLAGS


def main(_):
  pipeline = rs.pipeline()
  config = rs.config()
  config.enable_stream(rs.stream.color, 1280, 720,
                       rs.format.bgr8, FLAGS.frame_rate)
  pipeline.start(config)

  camera_image_publisher = rospy.Publisher('/perception/camera_image',
                                           Image,
                                           queue_size=1)
  rospy.init_node('realsense_camera_capture', anonymous=True)
  rate = rospy.Rate(10)
  while not rospy.is_shutdown():
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.array(color_frame.get_data())
    color_image = cv2.resize(color_image, dsize=(FLAGS.frame_width, FLAGS.frame_height))
    image = ros_numpy.msgify(Image,
                             cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB),
                             encoding='rgb8')
    camera_image_publisher.publish(image)
    rate.sleep()
  pipeline.stop()


if __name__ == "__main__":
  app.run(main)
