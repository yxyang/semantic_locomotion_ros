"""Train TF-based FCHarDNet Model."""
import os

from absl import app
from absl import flags

from ml_collections import config_flags
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import ray

from m1_perception.fchardnet.data_loader import rugd_data_loader
from m1_perception.fchardnet.model import HardNet

config_flags.DEFINE_config_file("config", "fchardnet/configs/rugd.py",
                                "Experiment config.")
FLAGS = flags.FLAGS


def main(argv):
  del argv  # unused

  config = FLAGS.config
  ray.init(address='auto', _redis_password='1234')
  data_loader_class = ray.remote(rugd_data_loader.RUGDDataLoader)
  train_data_loader = data_loader_class.remote(
      root_dir=config.data.data_path,
      batch_size=config.training.batch_size,
      augmentations=config.data.augmentations,
      split='train')
  num_train_batches = ray.get(train_data_loader.num_batches.remote())

  val_data_loader = data_loader_class.remote(
      root_dir=config.data.data_path,
      batch_size=config.training.batch_size,
      augmentations=[],
      split='val')
  num_val_batches = ray.get(val_data_loader.num_batches.remote())

  model = HardNet()
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
  optimizer = tf.keras.optimizers.Adam()
  step_count = 0

  writer = tf.summary.create_file_writer(config.logdir)
  writer.set_as_default()

  for epoch_id in range(config.training.num_epoches):
    next_batch = train_data_loader.next_batch.remote()
    for _ in tqdm(range(num_train_batches)):
      images, masks = ray.get(next_batch)
      next_batch = train_data_loader.next_batch.remote()
      with tf.GradientTape() as tape:
        pred_masks = model(images)
        loss = loss_object(masks, pred_masks)
      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))
      step_count += 1

      with tf.name_scope('train'):
        tf.summary.scalar('Loss', data=loss.numpy(), step=step_count)


    if epoch_id % config.training.eval_frequency == 0:
      val_loss = []
      percent_correct = []
      next_batch = val_data_loader.next_batch.remote()
      for _ in tqdm(range(num_val_batches)):
        images, masks = ray.get(next_batch)
        next_batch = val_data_loader.next_batch.remote()

        pred_masks = model(images)
        loss = loss_object(masks, pred_masks)
        val_loss.append(loss.numpy())
        num_correct = np.sum(masks == np.argmax(pred_masks, axis=-1))
        percent_correct.append(num_correct / np.prod(masks.shape))

      with tf.name_scope('validation'):
        tf.summary.scalar('Loss', data=np.mean(val_loss), step=step_count)
        tf.summary.scalar('Percent Correct',
                          data=np.mean(percent_correct),
                          step=step_count)

    if epoch_id % config.training.save_frequency == 0:
      model.save_weights(
          os.path.join(config.logdir, 'cp-{}.ckpt'.format(epoch_id)))


if __name__ == "__main__":
  app.run(main)
