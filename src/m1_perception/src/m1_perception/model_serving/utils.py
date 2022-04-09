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
