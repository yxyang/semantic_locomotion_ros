#!/usr/bin/env python
"""GRPC Client to generate semantic embedding."""
from absl import app
from absl import flags

import cv2
import grpc
import numpy as np
import rospy

import ros_numpy
from sensor_msgs.msg import CompressedImage, Image

from a1_interface.msg import speed_command
from m1_perception.model_serving import semantic_embedding_service_pb2_grpc
from m1_perception.model_serving import utils
from m1_perception.speed_model import mask_utils

flags.DEFINE_string('server_addr', '10.211.55.2', 'server address.')
flags.DEFINE_integer('port', 5005, 'port number.')
FLAGS = flags.FLAGS


class SpeedMapGenerator:
  """Generates speed map from camera inputs."""
  def __init__(self, grpc_stub):
    self._mask = mask_utils.get_segmentation_mask(width=424, height=240)
    self._grpc_stub = grpc_stub
    self._speed_command_publisher = rospy.Publisher('autospeed_command',
                                                    speed_command,
                                                    queue_size=1)
    self._speed_map_publisher = rospy.Publisher(
        '/perception/speedmap/image_raw', Image, queue_size=1)
    self._speed_visualization_publisher = rospy.Publisher(
        '/perception/speedmap/visualization/compressed',
        CompressedImage,
        queue_size=1)

  def camera_image_callback(self, image_msg):
    """Computes speed command and speed map from camera image."""
    image_array = ros_numpy.numpify(image_msg)
    image_array = image_array.astype(np.float32) / 255.
    image_request = utils.numpy_array_to_grpc_message(image_array)
    response = self._grpc_stub.GetSpeedEstimation(image_request)
    response_image = utils.grpc_message_to_numpy_array(response)[..., 0]
    desired_speed = np.sum(response_image * self._mask) / np.sum(self._mask)
    command = speed_command(vel_x=desired_speed,
                            vel_y=0,
                            rot_z=0,
                            timestamp=rospy.get_rostime())
    self._speed_command_publisher.publish(command)
    msg = ros_numpy.image.numpy_to_image(response_image.astype(np.float32),
                                         encoding='32FC1')
    msg.header.stamp = image_msg.header.stamp
    msg.header.frame_id = image_msg.header.frame_id
    self._speed_map_publisher.publish(msg)

    msg = CompressedImage()
    msg.header.stamp = rospy.Time.now()
    msg.format = "png"
    msg.data = np.array(
        cv2.imencode(".png", convert_to_rgb(response_image))[1]).tobytes()
    self._speed_visualization_publisher.publish(msg)


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
  # GRPC Settings
  options = [('grpc.max_send_message_length', 5 * 1024 * 1024),
             ('grpc.max_receive_message_length', 5 * 1024 * 1024)]
  channel = grpc.insecure_channel('{}:{}'.format(FLAGS.server_addr,
                                                 FLAGS.port),
                                  options=options)
  stub = semantic_embedding_service_pb2_grpc.SemanticEmbeddingStub(channel)
  speed_map_generator = SpeedMapGenerator(stub)
  rospy.Subscriber("/camera/color/image_raw", Image,
                   speed_map_generator.camera_image_callback)
  rospy.spin()


if __name__ == "__main__":
  app.run(main)
