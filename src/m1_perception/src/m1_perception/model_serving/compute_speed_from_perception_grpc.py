#!/usr/bin/env python
"""GRPC Client to generate semantic embedding."""
from absl import app
from absl import flags

import grpc
import numpy as np
import pyrealsense2 as rs
import rospy

from a1_interface.msg import speed_command
from m1_perception.model_serving import semantic_embedding_service_pb2_grpc
from m1_perception.model_serving import utils
from m1_perception.speed_model import mask_utils

flags.DEFINE_string('server_addr', '10.211.55.2', 'server address.')
flags.DEFINE_integer('port', 5005, 'port number.')
FLAGS = flags.FLAGS


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
  pipeline.start(config)
  mask = mask_utils.get_segmentation_mask(height=424, width=240)
  while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.array(color_frame.get_data()).astype(np.float32) / 255.
    image_request = utils.numpy_array_to_grpc_message(color_image)
    response = stub.GetSpeedEstimation(image_request)
    response_image = utils.grpc_message_to_numpy_array(response)[0]

    desired_speed = np.sum(response_image * mask) / np.sum(mask)
    command = speed_command(vel_x=desired_speed,
                            vel_y=0,
                            rot_z=0,
                            timestamp=rospy.get_rostime())
    speed_command_publisher.publish(command)


if __name__ == "__main__":
  app.run(main)
