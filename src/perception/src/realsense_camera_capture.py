#!/usr/bin/env python
"""Example for real-time semantic segmentation using realsense camera."""
import cv2
import numpy as np
import pyrealsense2 as rs
import rospy
from absl import app, flags
from sensor_msgs.msg import CompressedImage

flags.DEFINE_integer('frame_width', 640, 'frame width.')
flags.DEFINE_integer('frame_height', 360, 'frame height.')
flags.DEFINE_integer('frame_rate', 10, 'frame rate.')
FLAGS = flags.FLAGS


def main(_):
  pipeline = rs.pipeline()
  config = rs.config()
  config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8,
                       FLAGS.frame_rate)
  pipeline.start(config)

  camera_image_publisher = rospy.Publisher(
      '/perception/camera_image/compressed', CompressedImage, queue_size=1)
  rospy.init_node('realsense_camera_capture', anonymous=True)
  rate = rospy.Rate(FLAGS.frame_rate)
  while not rospy.is_shutdown():
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.array(color_frame.get_data())
    color_image = cv2.resize(color_image,
                             dsize=(FLAGS.frame_width, FLAGS.frame_height))
    # color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    msg = CompressedImage()
    msg.header.stamp = rospy.Time.now()
    msg.format = "png"
    msg.data = np.array(cv2.imencode(".png", color_image)[1]).tostring()
    camera_image_publisher.publish(msg)
    rate.sleep()
  pipeline.stop()


if __name__ == "__main__":
  app.run(main)
