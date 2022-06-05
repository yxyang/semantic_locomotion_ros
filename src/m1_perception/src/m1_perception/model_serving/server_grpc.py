#!/usr/bin/env python
"""GRPC server for semantic embedding."""
from concurrent import futures
import time

from absl import app
from absl import flags
from absl import logging

import grpc
import numpy as np

from m1_perception.fchardnet.model import HardNet
from m1_perception.model_serving import semantic_embedding_service_pb2_grpc
from m1_perception.model_serving import utils
from m1_perception.speed_model.model import SpeedModel
from m1_perception.speed_model import mask_utils

flags.DEFINE_string("vision_model_dir",
                    "m1_perception/checkpoints/vision_model/cp-99.ckpt",
                    "path to model checkpoint.")
flags.DEFINE_string('speed_model_dir',
                    'm1_perception/checkpoints/speed_model/cp-99.ckpt',
                    'path to speed model.')
flags.DEFINE_string('server_addr', '10.211.55.2', 'server address.')
flags.DEFINE_integer('port', 5005, 'port number.')
FLAGS = flags.FLAGS


class SemanticEmbeddingServicer(
    semantic_embedding_service_pb2_grpc.SemanticEmbeddingServicer):
  """GRPC service for generating semantic embedding and estimating speed."""
  def __init__(self, model_dir, speed_model_dir):
    self._vision_model = HardNet()
    self._vision_model.load_weights(model_dir)
    self._speed_model = SpeedModel()
    self._speed_model.load_weights(speed_model_dir)
    self._mask, self._mask_size = None, None

  def GetSpeedEstimation(self, request, context):
    """Returns speed estimation given camera image."""
    del context  # unused
    if (self._mask is None) or (request.height != self._mask_size[0]) or (
        request.width != self._mask_size[1]):
      self._mask_size = (request.height, request.width)
      self._mask = mask_utils.get_segmentation_mask(height=request.height,
                                                    width=request.width)
    imgarr = utils.grpc_message_to_numpy_array(request)[np.newaxis, ...]
    semantic_embedding = self._vision_model.get_embedding(imgarr)[0]
    pred_speed = self._speed_model(semantic_embedding).numpy()[..., np.newaxis]
    return utils.numpy_array_to_grpc_message(pred_speed)

  def GetSemanticSegmentation(self, request, context):
    del context  # unused
    if (not self._mask) or (request.height != self._mask_size[0]) or (
        request.width != self._mask_size[1]):
      self._mask_size = (request.height, request.width)
      self._mask = mask_utils.get_segmentation_mask(height=request.height,
                                                    width=request.width)
    imgarr = utils.grpc_message_to_numpy_array(request)[np.newaxis, ...]
    semantic_segmentation = self._vision_model(imgarr)[0].numpy()
    return utils.numpy_array_to_grpc_message(semantic_segmentation)


def main(argv):
  del argv  # unused
  options = [('grpc.max_send_message_length', 5 * 1024 * 1024),
             ('grpc.max_receive_message_length', 5 * 1024 * 1024)]
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=12),
                       options=options)
  servicer = SemanticEmbeddingServicer(FLAGS.vision_model_dir,
                                       FLAGS.speed_model_dir)
  semantic_embedding_service_pb2_grpc.add_SemanticEmbeddingServicer_to_server(
      servicer, server)

  server.add_insecure_port('{}:{}'.format(FLAGS.server_addr, FLAGS.port))
  server.start()

  logging.info("Server running...")
  try:
    while True:
      time.sleep(5)
  except KeyboardInterrupt:
    server.stop(0)


if __name__ == "__main__":
  app.run(main)
