"""Train TF-based FCHarDNet Model."""
import os

from absl import app
from absl import flags

from ml_collections import config_flags
import numpy as np
import tensorflow as tf
from tqdm import tqdm
# import ray

from m1_perception.speed_model_nopretrain.data_loader import data_loader
from m1_perception.speed_model_nopretrain.model import HardNet
from m1_perception.speed_model import mask_utils

config_flags.DEFINE_config_file(
    "config", "m1_perception/speed_model_nopretrain/configs/a1.py",
    "Experiment config.")
FLAGS = flags.FLAGS


def main(argv):
  del argv  # unused
  config = FLAGS.config
  # ray.init(address='auto', _redis_password='1234')
  data_loader_class = data_loader.DataLoader
  train_data_loader = data_loader_class(
      root_dir=config.data.train_data_path,
      batch_size=config.training.batch_size,
      augmentations=config.data.augmentations)
  num_train_batches = train_data_loader.num_batches

  val_data_loader = data_loader_class(root_dir=config.data.val_data_path,
                                      batch_size=config.training.batch_size,
                                      augmentations=[])
  num_val_batches = val_data_loader.num_batches

  images, _ = train_data_loader.next_batch()
  height, width = images[0].shape[0], images[0].shape[1]
  mask = mask_utils.get_segmentation_mask(height=height, width=width)

  model = HardNet()
  loss_object = tf.keras.losses.MeanSquaredError()
  optimizer = tf.keras.optimizers.Adam()
  step_count = 0

  writer = tf.summary.create_file_writer(config.logdir)
  writer.set_as_default()

  for epoch_id in range(config.training.num_epoches):
    next_batch = train_data_loader.next_batch()
    for _ in tqdm(range(num_train_batches)):
      images, speeds = next_batch
      next_batch = train_data_loader.next_batch()
      with tf.GradientTape() as tape:
        speed_map = model(images)
        pred_speed = tf.reduce_sum(speed_map * mask,
                                   axis=(1, 2)) / np.sum(mask)
        loss = loss_object(pred_speed, speeds)
      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))
      step_count += 1

      with tf.name_scope('train'):
        tf.summary.scalar('Loss', data=loss.numpy(), step=step_count)

    if epoch_id % config.training.eval_frequency == 0:
      val_loss = []
      next_batch = val_data_loader.next_batch()
      for _ in tqdm(range(num_val_batches)):
        images, speeds = next_batch
        next_batch = val_data_loader.next_batch()
        speed_map = model(images)
        pred_speed = tf.reduce_sum(speed_map * mask,
                                   axis=(1, 2)) / np.sum(mask)
        loss = loss_object(pred_speed, speeds)
        val_loss.append(loss.numpy())
      with tf.name_scope('validation'):
        tf.summary.scalar('Loss', data=np.mean(val_loss), step=step_count)

    if epoch_id % config.training.save_frequency == 0:
      model.save_weights(
          os.path.join(config.logdir, 'cp-{}.ckpt'.format(epoch_id)))


if __name__ == "__main__":
  app.run(main)
