"""Train speed model."""
import os

from absl import app
from absl import flags

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from m1_perception.speed_model import data_loader
from m1_perception.speed_model.model import SpeedModel
from m1_perception.speed_model import mask_utils

flags.DEFINE_string('train_data_dir', 'speed_model/data/train',
                    'path to training data.')
flags.DEFINE_string('val_data_dir', 'speed_model/data/val',
                    'path to validation data.')
flags.DEFINE_string('logdir', 'speed_model_logs', 'path to logging directory.')
flags.DEFINE_integer('batch_size', 6, 'batch size.')
flags.DEFINE_integer('num_epoches', 100, 'number of epoches.')
flags.DEFINE_integer('save_frequency', 1, 'logging frequency.')
flags.DEFINE_integer('eval_frequency', 1, 'eval frequency.')
FLAGS = flags.FLAGS


def main(argv):
  del argv  # unused
  train_loader = data_loader.DataLoader(FLAGS.train_data_dir, FLAGS.batch_size)
  val_loader = data_loader.DataLoader(FLAGS.val_data_dir, FLAGS.batch_size)

  model = SpeedModel(num_hidden_layers=1, dim_hidden=20)
  loss_object = tf.keras.losses.MeanSquaredError()
  optimizer = tf.keras.optimizers.Adam()

  embeddings, _ = train_loader.next_batch()
  height, width = embeddings[0].shape[0], embeddings[0].shape[1]
  mask = mask_utils.get_segmentation_mask(height=height, width=width)

  if not os.path.exists(FLAGS.logdir):
    os.makedirs(FLAGS.logdir)
  writer = tf.summary.create_file_writer(FLAGS.logdir)
  writer.set_as_default()

  step_count = 0
  for epoch_id in range(1000):
    for _ in tqdm(range(train_loader.num_batches)):
      embeddings, actual_speed = train_loader.next_batch()
      with tf.GradientTape() as tape:
        pred_speed = model(embeddings)
        pred_speed = tf.reduce_sum(pred_speed * mask,
                                   axis=(1, 2)) / np.sum(mask)
        loss = loss_object(pred_speed, actual_speed)
      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))
      step_count += 1
      with tf.name_scope('train'):
        tf.summary.scalar('Loss', data=loss.numpy(), step=step_count)

    if epoch_id % FLAGS.eval_frequency == 0:
      val_losses = []
      for _ in tqdm(range(val_loader.num_batches)):
        embeddings, actual_speed = val_loader.next_batch()
        pred_speed = model(embeddings)
        pred_speed = tf.reduce_sum(pred_speed * mask,
                                   axis=(1, 2)) / np.sum(mask)
        loss = loss_object(pred_speed, actual_speed)
        val_losses.append(loss.numpy())

      with tf.name_scope('validation'):
        tf.summary.scalar('Loss', data=np.mean(val_losses), step=step_count)

    if epoch_id % FLAGS.save_frequency == 0:
      model.save_weights(
          os.path.join(FLAGS.logdir, 'cp-{}.ckpt'.format(epoch_id)))


if __name__ == "__main__":
  app.run(main)
