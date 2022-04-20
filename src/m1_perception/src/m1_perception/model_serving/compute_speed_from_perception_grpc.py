#!/usr/bin/env python
"""GRPC Client to generate semantic embedding."""
from absl import app
from absl import flags

import cv2
import grpc
import numpy as np
import pyrealsense2 as rs
import rospy
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import CompressedImage, Image

from a1_interface.msg import speed_command
from m1_perception.model_serving import semantic_embedding_service_pb2_grpc
from m1_perception.model_serving import utils
from m1_perception.speed_model import mask_utils

flags.DEFINE_string('server_addr', '10.211.55.2', 'server address.')
flags.DEFINE_integer('port', 5005, 'port number.')
flags.DEFINE_bool(
    'publish_ros_topic', True,
    'whether to publish camera image and motion data as a ros topic.')
FLAGS = flags.FLAGS


def convert_to_rgb(speed_map, min_speed=0, max_speed=2):
  """Converts a HxW speedmap into HxWx3 RGB image for visualization."""
  speed_map = np.clip(speed_map, min_speed, max_speed)
  # Interpolate between 0 and 1
  speed_map = (speed_map - min_speed) / (max_speed - min_speed)
  slow_color = np.array([0, 0, 255.])
  fast_color = np.array([0, 255., 0])

  channels = []
  for channel_id in range(3):
    channel_value = slow_color[channel_id] * (
        1 - speed_map) + fast_color[channel_id] * speed_map
    channels.append(channel_value)

  return np.stack(channels, axis=-1)


def main(argv):
  del argv  # unused
  rospy.init_node('speed_from_perception', anonymous=True)
  speed_command_publisher = rospy.Publisher('autospeed_command',
                                            speed_command,
                                            queue_size=1)

  # GRPC Settings
  options = [('grpc.max_send_message_length', 5 * 1024 * 1024),
             ('grpc.max_receive_message_length', 5 * 1024 * 1024)]
  channel = grpc.insecure_channel('{}:{}'.format(FLAGS.server_addr,
                                                 FLAGS.port),
                                  options=options)
  stub = semantic_embedding_service_pb2_grpc.SemanticEmbeddingStub(channel)

  pipeline = rs.pipeline()
  config = rs.config()
  config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 60)
  config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
  pipeline.start(config)
  mask = mask_utils.get_segmentation_mask(width=424, height=240)

  if FLAGS.publish_ros_topic:
    camera_image_publisher = rospy.Publisher(
        '/perception/camera_image_color/compressed',
        CompressedImage,
        queue_size=1)
    depth_image_publisher = rospy.Publisher(
        '/perception/camera_image_depth',
        Image,
        queue_size=1)
    speed_map_publisher = rospy.Publisher(
        '/perception/speed_map_2d/compressed', CompressedImage, queue_size=1)
    camera_orientation_publisher = rospy.Publisher(
        '/perception/camera_orientation', Quaternion, queue_size=1)

  while not rospy.is_shutdown():
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.array(color_frame.get_data()).astype(np.float32) / 255.
    image_request = utils.numpy_array_to_grpc_message(color_image)
    response = stub.GetSpeedEstimation(image_request)
    response_image = utils.grpc_message_to_numpy_array(response)[..., 0]
    desired_speed = np.sum(response_image * mask) / np.sum(mask)
    command = speed_command(vel_x=desired_speed,
                            vel_y=0,
                            rot_z=0,
                            timestamp=rospy.get_rostime())
    speed_command_publisher.publish(command)

    depth_frame = frames.get_depth_frame()
    if FLAGS.publish_ros_topic:
      # msg = CompressedImage()
      # msg.header.stamp = rospy.Time.now()
      # msg.format = "png"
      # msg.data = np.array(
      #     cv2.imencode(".png",
      #                  np.array(depth_frame.get_data()))[1]).tostring()
      msg = Image()
      msg.header.stamp = rospy.Time.now()
      msg.header.frame_id = 'camera'
      msg.height = 480
      msg.width = 640
      msg.encoding = 'mono16'
      # msg.format = "png"
      msg.data = np.array(depth_frame.get_data()).tostring()
      depth_image_publisher.publish(msg)

      msg = CompressedImage()
      msg.header.stamp = rospy.Time.now()
      msg.format = "png"
      msg.data = np.array(
          cv2.imencode(".png",
                       np.array(color_frame.get_data()))[1]).tostring()
      camera_image_publisher.publish(msg)

      msg = CompressedImage()
      msg.header.stamp = rospy.Time.now()
      msg.format = "png"
      speed_map_rgb = convert_to_rgb(response_image)
      msg.data = np.array(cv2.imencode(".png", speed_map_rgb)[1]).tostring()
      speed_map_publisher.publish(msg)

      msg = Quaternion()
      msg.x, msg.y, msg.z, msg.w = 0, 0, 0, 1
      camera_orientation_publisher.publish(msg)


if __name__ == "__main__":
  app.run(main)
