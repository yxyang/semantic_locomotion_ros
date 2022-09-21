"""Example of running the speed model."""
from absl import app
from absl import flags

import numpy as np

from m1_perception.fchardnet.model import HardNet
from m1_perception.speed_model_class_category.model import SpeedModel

FLAGS = flags.FLAGS


def main(argv):
  del argv  # unused
  model = HardNet()
  model.load_weights('m1_perception/checkpoints/vision_model/cp-99.ckpt')
  speed_model = SpeedModel(num_hidden_layers=1, dim_hidden=20)
  # speed_model.load_weights('speed_model_logs/trial_1/cp-60.ckpt')
  test_image = np.random.random(size=(1, 640, 480, 3))
  embedding = model.get_prediction_onehot(test_image)
  speed = speed_model(embedding)
  print(speed.numpy().shape)


if __name__ == "__main__":
  app.run(main)
