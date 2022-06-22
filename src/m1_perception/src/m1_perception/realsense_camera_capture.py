#!/usr/bin/env python
"""Example for real-time semantic segmentation using realsense camera."""
from absl import app
from absl import flags

import grpc
import numpy as np
import pyrealsense2 as rs
import rospy

import ros_numpy
from sensor_msgs.msg import Image, PointCloud2

from a1_interface.msg import speed_command
from m1_perception.model_serving import semantic_embedding_service_pb2_grpc
from m1_perception.model_serving import utils
from m1_perception.speed_model import mask_utils

flags.DEFINE_string('grpc_server_addr', '10.211.55.2', 'server address.')
flags.DEFINE_integer('port', 5005, 'port number.')
flags.DEFINE_integer('frame_width', 424, 'frame width.')
flags.DEFINE_integer('frame_height', 240, 'frame height.')
flags.DEFINE_integer('frame_rate', 3, 'frame rate.')

# Topic settings
flags.DEFINE_bool('publish_color_image', True,
                  'whether to publish color image.')
flags.DEFINE_bool('publish_depth_image', True,
                  'whether to publish depth image.')
flags.DEFINE_bool('publish_color_pointcloud', False,
                  'whether to publish color pointcloud.')
flags.DEFINE_bool('publish_speedmap_image', True,
                  'whether to publish speedmap image')
flags.DEFINE_bool('publish_speedmap_pointcloud', True,
                  'whether to publish speedmap pointcloud')
flags.DEFINE_bool('publish_speed_command', True,
                  'whether to publish speedmap pointcloud')
FLAGS = flags.FLAGS


def compute_speed_map(image_array_bgr, stub, segmentation_mask):
  """Queries GRPC stub to compute speed map."""
  image_array_rgb = image_array_bgr[..., ::-1]
  image_array_rgb = image_array_rgb.astype(np.float32) / 255.
  image_request = utils.numpy_array_to_grpc_message(image_array_rgb)
  response = stub.GetSpeedEstimation(image_request)
  response_image = utils.grpc_message_to_numpy_array(response)[..., 0]
  response_image = np.clip(response_image, 0, 2)
  desired_speed = np.sum(
      response_image * segmentation_mask) / np.sum(segmentation_mask)
  desired_speed = np.clip(desired_speed * 1.15, 0.5, 2)
  command = speed_command(vel_x=desired_speed,
                          vel_y=0,
                          rot_z=0,
                          timestamp=rospy.get_rostime())
  return response_image, command


def main(_):
  rospy.init_node('realsense_camera_capture', anonymous=True)

  # GRPC Settings
  options = [('grpc.max_send_message_length', 5 * 1024 * 1024),
             ('grpc.max_receive_message_length', 5 * 1024 * 1024)]
  channel = grpc.insecure_channel('{}:{}'.format(FLAGS.grpc_server_addr,
                                                 FLAGS.port),
                                  options=options)
  stub = semantic_embedding_service_pb2_grpc.SemanticEmbeddingStub(channel)
  segmentation_mask = mask_utils.get_segmentation_mask(width=424, height=240)

  # Realsense Settings
  pipeline = rs.pipeline()
  config = rs.config()
  config.enable_stream(rs.stream.color, FLAGS.frame_width, FLAGS.frame_height,
                       rs.format.bgr8, 60)
  config.enable_stream(rs.stream.depth, FLAGS.frame_width, FLAGS.frame_height,
                       rs.format.z16, 60)
  pipeline.start(config)
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
    color_pointcloud_publisher = rospy.Publisher('/camera/color/pointcloud',
                                                 PointCloud2,
                                                 queue_size=1)

  if FLAGS.publish_speedmap_image:
    speedmap_image_publisher = rospy.Publisher(
        '/perception/speedmap/image_raw', Image, queue_size=1)

  if FLAGS.publish_speedmap_pointcloud:
    speedmap_pointcloud_publisher = rospy.Publisher(
        '/perception/speedmap/pointcloud', PointCloud2, queue_size=1)

  if FLAGS.publish_speed_command:
    speed_command_publisher = rospy.Publisher('autospeed_command',
                                              speed_command,
                                              queue_size=1)

  rate = rospy.Rate(FLAGS.frame_rate)
  while not rospy.is_shutdown():
    # Get frame
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not depth_frame or not color_frame:
      continue

    # Process data
    color_image = np.array(color_frame.get_data())
    depth_image = np.array(depth_frame.get_data())
    points = pointcloud.calculate(depth_frame)
    pointcloud.map_to(color_frame)
    vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(
        color_image.shape)
    speedmap_image, speed_command_msg = compute_speed_map(
        color_image, stub, segmentation_mask)
    speed_image_bgr = utils.convert_speedmap_to_bgr(speedmap_image)

    # Publish messages
    if FLAGS.publish_color_image:
      msg = ros_numpy.image.numpy_to_image(color_image, 'bgr8')
      msg.header.frame_id = 'camera_link'
      color_image_publisher.publish(msg)

    if FLAGS.publish_depth_image:
      msg = ros_numpy.image.numpy_to_image(depth_image, 'mono16')
      msg.header.frame_id = 'camera_link'
      depth_image_publisher.publish(msg)

    if FLAGS.publish_color_pointcloud:
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

    if FLAGS.publish_speedmap_image:
      msg = ros_numpy.image.numpy_to_image(speed_image_bgr, 'bgr8')
      msg.header.frame_id = 'camera_link'
      speedmap_image_publisher.publish(msg)

    if FLAGS.publish_speedmap_pointcloud:
      dtype = np.dtype({
          'names': ['speed', 'b', 'g', 'r', 'x', 'y', 'z'],
          'formats': [
              np.float32, np.float32, np.float32, np.float32, np.float32,
              np.float32, np.float32
          ]
      })
      arr = np.concatenate((
          speedmap_image[..., np.newaxis],
          speed_image_bgr / 255.,
          vertices,
      ),
                           axis=2).astype(np.float32).view(dtype=dtype)
      msg = ros_numpy.point_cloud2.array_to_pointcloud2(arr,
                                                        frame_id='camera_link')
      speedmap_pointcloud_publisher.publish(msg)

    if FLAGS.publish_speed_command:
      speed_command_publisher.publish(speed_command_msg)

    rate.sleep()

  pipeline.stop()


if __name__ == "__main__":
  app.run(main)
