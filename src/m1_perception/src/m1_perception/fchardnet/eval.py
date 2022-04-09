"""Evaluates TF-based FCHarDNet Model"""

from absl import app
from absl import flags

from ml_collections import config_flags
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from m1_perception.fchardnet.data_loader import rugd_data_loader
from m1_perception.fchardnet.model import HardNet

flags.DEFINE_string('model_dir', None, 'path to saved model.')
config_flags.DEFINE_config_file("config", "fchardnet/configs/rugd.py",
                                "Experiment config.")
FLAGS = flags.FLAGS


def main(argv):
  del argv  # unused

  model = HardNet()
  model.load_weights(FLAGS.model_dir)
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

  config = FLAGS.config
  data_loader_class = rugd_data_loader.RUGDDataLoader
  val_data_loader = data_loader_class(root_dir=config.data.data_path,
                                      batch_size=config.training.batch_size,
                                      augmentations=[],
                                      split='test')
  num_val_batches = val_data_loader.num_batches()

  val_loss = []
  percent_correct = []
  for _ in tqdm(range(num_val_batches)):
    images, masks = val_data_loader.next_batch()
    pred_masks = model(images)
    loss = loss_object(masks, pred_masks)
    val_loss.append(loss.numpy())
    num_correct = np.sum(masks == np.argmax(pred_masks, axis=-1))
    percent_correct.append(num_correct / np.prod(masks.shape))

  print("Accuracy: {}".format(np.mean(percent_correct)))
  print("Loss: {}".format(np.mean(val_loss)))


if __name__ == "__main__":
  app.run(main)
