"""Example of running the speed model."""
from absl import app
from absl import flags

import numpy as np

from m1_perception.fchardnet.model import HardNet
from m1_perception.speed_model.speed_model import SpeedModel

FLAGS = flags.FLAGS


def main(argv):
  del argv  # unused
  model = HardNet()
  model.load_weights('logs/cp-99.ckpt')
  model.load_pca_data('logs/pca_data.npz')
  speed_model = SpeedModel(num_hidden_layers=1, dim_hidden=20)
  test_image = np.random.random(size=(1, 640, 480, 3))
  embedding = model.get_embedding_lowdim(test_image)
  speed = speed_model(embedding)
  print(speed.numpy().shape)


if __name__ == "__main__":
  app.run(main)
