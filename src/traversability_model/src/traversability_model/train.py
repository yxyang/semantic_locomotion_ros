#!/usr/bin/env python
"""Trains models to predict traversability and select gaits."""
import os

from absl import app
from absl import flags

from ml_collections.config_flags import config_flags
import numpy as np
from sklearn.decomposition import PCA
import torch
from torch.utils.data import TensorDataset

config_flags.DEFINE_config_file(
    'config',
    'src/traversability_model/src/traversability_model/configs/config_human.py'
)
FLAGS = flags.FLAGS


def moving_average(data, window_size=60):
  ans = np.zeros_like(data)
  for idx in range(data.shape[0]):
    start_idx = np.maximum(0, idx - window_size)
    ans[idx] = np.mean(data[start_idx:start_idx + window_size], axis=0)
  return ans


def construct_and_train_model(config, inputs, labels, gaits, desired_gait=0):
  """Construct the model and train it according to config."""
  indices = np.nonzero(gaits == desired_gait)[0]
  selected_inputs, selected_labels = inputs[indices], labels[indices]

  device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
      'cpu')
  tensor_x = torch.Tensor(selected_inputs).to(device)
  tensor_y = torch.Tensor(selected_labels).reshape((-1, 1)).to(device)
  dataset = TensorDataset(tensor_x, tensor_y)
  train_size = int(0.85 * len(dataset))
  val_size = len(dataset) - train_size
  train_set, val_set = torch.utils.data.random_split(dataset,
                                                     [train_size, val_size])

  model = config.model_class(dim_in=config.model.dim_in,
                             dim_out=config.model.dim_out,
                             num_hidden=config.model.num_hidden,
                             dim_hidden=config.model.dim_hidden).to(device)
  model.train(train_set,
              val_set,
              num_epochs=config.model_training.num_epochs[desired_gait],
              batch_size=config.model_training.batch_size)
  return model


def main(argv):
  del argv  # unused
  config = FLAGS.config

  # Load data and generate mdoel input / labels
  processed_data = dict(np.load(open(config.data_dir, 'rb'),
                                allow_pickle=True))
  smoothed_image_embeddings = moving_average(
      processed_data['embeddings'], config.feature_moving_average_window_size)
  pca = PCA(n_components=config.pca_output_dim)
  pca.fit(smoothed_image_embeddings)
  inputs = pca.transform(smoothed_image_embeddings)
  labels = config.label_generator(processed_data)

  model_crawl = construct_and_train_model(config,
                                          inputs,
                                          labels,
                                          processed_data['gaits'],
                                          desired_gait=0)

  model_walk = construct_and_train_model(config,
                                         inputs,
                                         labels,
                                         processed_data['gaits'],
                                         desired_gait=1)

  model_run = construct_and_train_model(config,
                                        inputs,
                                        labels,
                                        processed_data['gaits'],
                                        desired_gait=2)

  state = dict(model_crawl=model_crawl.state_dict(),
               model_walk=model_walk.state_dict(),
               model_run=model_run.state_dict())
  save_path = os.path.join(config.output_dir, 'trained_model.pkl')
  torch.save(state, save_path)

  save_path = os.path.join(config.output_dir, 'pca_training_data.npz')
  np.savez(save_path, pca_train_data=smoothed_image_embeddings)


if __name__ == "__main__":
  app.run(main)
