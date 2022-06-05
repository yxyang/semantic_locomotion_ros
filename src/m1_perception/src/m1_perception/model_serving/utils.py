"""Utilities for converting between np array and gRPC messages."""
import base64

import numpy as np

from m1_perception.model_serving import semantic_embedding_service_pb2


def numpy_array_to_grpc_message(array):
  return semantic_embedding_service_pb2.B64Image(
      b64image=base64.b64encode(array.flatten()),
      width=array.shape[0],
      height=array.shape[1],
      num_channels=array.shape[2])


def grpc_message_to_numpy_array(message):
  b64decoded = base64.b64decode(message.b64image)
  return np.frombuffer(b64decoded,
                       dtype=np.float32).reshape(message.width,
                                                 message.height,
                                                 message.num_channels)

def convert_speedmap_to_bgr(speed_map, min_speed=0, max_speed=2):
  """Converts a HxW speedmap into HxWx3 RGB image for visualization."""
  speed_map = np.clip(speed_map, min_speed, max_speed)
  # Interpolate between 0 and 1
  speed_map = 2 * (speed_map - min_speed) / (max_speed - min_speed)
  slow_color = np.array([0, 0, 255.])
  fast_color = np.array([0, 255., 0])

  channels = []
  for channel_id in range(3):
    channel_value = slow_color[channel_id] * (
        2 - speed_map) + fast_color[channel_id] * speed_map
    channel_value = np.clip(channel_value, 0, 255.)
    channels.append(channel_value)
  return np.stack(channels, axis=-1).astype(np.uint8)
