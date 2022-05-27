#!/usr/bin/env python
"""Example for real-time semantic segmentation using realsense camera."""
from absl import app
from absl import flags

import numpy as np
import pyrealsense2 as rs
import rospy

import ros_numpy
from sensor_msgs.msg import Image, PointCloud2

flags.DEFINE_integer('frame_width', 424, 'frame width.')
flags.DEFINE_integer('frame_height', 240, 'frame height.')
flags.DEFINE_integer('frame_rate', 6, 'frame rate.')
flags.DEFINE_bool('publish_color_image', True,
                  'whether to publish color image.')
flags.DEFINE_bool('publish_depth_image', True,
                  'whether to publish depth image.')
flags.DEFINE_bool('publish_color_pointcloud', True,
                  'whether to publish color pointcloud.')
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
  pointcloud = rs.pointcloud()

  if FLAGS.publish_color_image:
    color_image_publisher = rospy.Publisher('/camera/color/image_raw',
                                            Image,
                                            queue_size=1)
  if FLAGS.publish_depth_image:
    depth_image_publisher = rospy.Publisher(
        '/camera/aligned_depth_to_color/image_raw', Image, queue_size=1)

  if FLAGS.publish_color_pointcloud:
    color_pointcloud_publisher = rospy.Publisher('/camera/color_pointcloud',
                                                 PointCloud2,
                                                 queue_size=1)

  rate = rospy.Rate(FLAGS.frame_rate)
  while not rospy.is_shutdown():
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not depth_frame or not color_frame:
      continue

    # Publish messages
    color_image = np.array(color_frame.get_data())
    depth_image = np.array(depth_frame.get_data())

    if FLAGS.publish_color_image:
      msg = ros_numpy.image.numpy_to_image(color_image, 'bgr8')
      msg.header.frame_id = 'camera_link'
      color_image_publisher.publish(msg)

    if FLAGS.publish_depth_image:
      msg = ros_numpy.image.numpy_to_image(depth_image, 'mono16')
      msg.header.frame_id = 'camera_link'
      depth_image_publisher.publish(msg)

    if FLAGS.publish_color_pointcloud:
      # Process frames
      points = pointcloud.calculate(depth_frame)
      pointcloud.map_to(color_frame)
      vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(
          color_image.shape)
      dtype = np.dtype({
          'names': ['b', 'g', 'r', 'x', 'y', 'z'],
          'formats': [
              np.float32, np.float32, np.float32, np.float32, np.float32,
              np.float32
          ]
      })
      arr = np.concatenate((color_image / 255., vertices),
                           axis=2).astype(np.float32).view(dtype=dtype)
      msg = ros_numpy.point_cloud2.array_to_pointcloud2(arr,
                                                        frame_id='camera_link')
      color_pointcloud_publisher.publish(msg)

    rate.sleep()

  pipeline.stop()


if __name__ == "__main__":
  app.run(main)
