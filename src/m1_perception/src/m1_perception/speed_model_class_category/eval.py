"""Evaluate speed model."""
import os

from absl import app
from absl import flags

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from m1_perception.speed_model_class_category import data_loader
from m1_perception.speed_model_class_category.model import SpeedModel
from m1_perception.speed_model import mask_utils

flags.DEFINE_string(
    'val_data_dir',
    'm1_perception/speed_model/data/ghost_memory_taylor_demo/val',
    'path to validation data.')
flags.DEFINE_string('logdir', 'speed_model_logs', 'path to logging directory.')
flags.DEFINE_integer('batch_size', 32, 'batch size.')
flags.DEFINE_integer('num_epoches', 60, 'number of epoches.')
FLAGS = flags.FLAGS


def main(argv):
  del argv  # unused
  val_loader = data_loader.DataLoader(FLAGS.val_data_dir, FLAGS.batch_size)

  model = SpeedModel(num_hidden_layers=1, dim_hidden=20)
  loss_object = tf.keras.losses.MeanSquaredError()

  embeddings, _ = val_loader.next_batch()
  height, width = embeddings[0].shape[0], embeddings[0].shape[1]
  mask = mask_utils.get_segmentation_mask(height=height, width=width)

  losses = []

  for epoch_id in tqdm(range(FLAGS.num_epoches)):
    model.load_weights(
        os.path.join(FLAGS.logdir, 'cp-{}.ckpt'.format(epoch_id)))
    val_losses = []
    for _ in range(val_loader.num_batches):
      embeddings, actual_speed = val_loader.next_batch()
      pred_speed = model(embeddings)
      pred_speed = tf.reduce_sum(pred_speed * mask, axis=(1, 2)) / np.sum(mask)
      loss = loss_object(pred_speed, actual_speed)
      val_losses.append(loss.numpy())

    losses.append(np.mean(val_losses))
    print(np.mean(val_losses))

  np.savez(open(os.path.join(FLAGS.logdir, 'eval_losses.npz'), 'wb'),
           losses=losses)


if __name__ == "__main__":
  app.run(main)
