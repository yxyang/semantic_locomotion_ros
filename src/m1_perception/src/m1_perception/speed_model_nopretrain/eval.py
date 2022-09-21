"""Evaluates TF-based FCHarDNet Model"""
import os

from absl import app
from absl import flags

from ml_collections import config_flags
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from m1_perception.speed_model_nopretrain.data_loader import data_loader
from m1_perception.speed_model_nopretrain.model import HardNet
from m1_perception.speed_model import mask_utils

flags.DEFINE_string('logdir', None, 'path to saved model.')
config_flags.DEFINE_config_file(
    "config", "m1_perception/speed_model_nopretrain/configs/a1.py",
    "Experiment config.")
flags.DEFINE_integer('num_epoches', 60, 'number of epoches.')
FLAGS = flags.FLAGS


def main(argv):
  del argv  # unused

  model = HardNet()
  loss_object = tf.keras.losses.MeanSquaredError()

  config = FLAGS.config
  data_loader_class = data_loader.DataLoader
  val_data_loader = data_loader_class(root_dir=config.data.val_data_path,
                                      batch_size=config.training.batch_size,
                                      augmentations=[])
  num_val_batches = val_data_loader.num_batches
  images, _ = val_data_loader.next_batch()
  height, width = images[0].shape[0], images[0].shape[1]
  mask = mask_utils.get_segmentation_mask(height=height, width=width)

  losses = []
  for epoch_id in tqdm(range(FLAGS.num_epoches)):
    model.load_weights(
        os.path.join(FLAGS.logdir, 'cp-{}.ckpt'.format(epoch_id)))
    val_losses = []
    for _ in range(num_val_batches):
      images, speeds = val_data_loader.next_batch()
      speed_map = model(images)
      pred_speed = tf.reduce_sum(speed_map * mask, axis=(1, 2)) / np.sum(mask)
      loss = loss_object(pred_speed, speeds)
      val_losses.append(loss.numpy())

    print("Loss: {}".format(np.mean(val_losses)))
    losses.append(np.mean(val_losses))

  np.savez(open(os.path.join(FLAGS.logdir, 'eval_losses.npz'), 'wb'),
           losses=losses)


if __name__ == "__main__":
  app.run(main)
