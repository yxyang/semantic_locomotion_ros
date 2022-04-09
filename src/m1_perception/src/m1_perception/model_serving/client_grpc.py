"""GRPC Client to generate semantic embedding."""
from absl import app
from absl import flags

import grpc
import numpy as np
from tqdm import tqdm

from m1_perception.model_serving import semantic_embedding_service_pb2_grpc
from m1_perception.model_serving import utils
flags.DEFINE_string('server_addr', '10.211.55.2', 'server address.')
flags.DEFINE_integer('port', 5005, 'port number.')
FLAGS = flags.FLAGS


def main(argv):
  del argv  # unused
  options = [('grpc.max_send_message_length', 5 * 1024 * 1024),
             ('grpc.max_receive_message_length', 5 * 1024 * 1024)]
  channel = grpc.insecure_channel('{}:{}'.format(FLAGS.server_addr,
                                                 FLAGS.port),
                                  options=options)
  stub = semantic_embedding_service_pb2_grpc.SemanticEmbeddingStub(channel)

  for _ in tqdm(range(100)):
    input_image = np.random.random(size=(640, 480, 3)).astype(np.float32)
    image_request = utils.numpy_array_to_grpc_message(input_image)
    response = stub.GetSpeedEstimation(image_request)
    response_image = utils.grpc_message_to_numpy_array(response)
    del response_image # unused

if __name__ == "__main__":
  app.run(main)
