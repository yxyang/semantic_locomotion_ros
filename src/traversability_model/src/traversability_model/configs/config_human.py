"""Configs for generating labels from human ratio."""

import ml_collections
import numpy as np

from traversability_model.models import probabilistic_model


def label_generator(data):
  labels = data['actual_speeds'][:, 0]
  labels -= np.mean(labels)
  labels /= np.std(labels)
  return labels


def get_config():
  """Configuration for using human command as label."""
  config = ml_collections.ConfigDict()

  # Model inputs / outputs
  config.data_dir = ('src/traversability_model/src/traversability_model/data/'
                     'processed_data.npz')
  config.feature_moving_average_window_size = 5
  config.pca_output_dim = 5
  config.label_generator = label_generator

  # Model setup
  config.model_class = probabilistic_model.ProbabilisticModel
  config.model = ml_collections.ConfigDict()
  config.model.dim_in = 5
  config.model.dim_out = 1
  config.model.num_hidden = 1
  config.model.dim_hidden = 20

  # Training setup
  config.model_training = ml_collections.ConfigDict()
  config.model_training.batch_size = 400
  config.model_training.num_epochs = [250, 125, 1500]

  # Output dir
  config.output_dir = ('src/traversability_model/src/traversability_model/'
                       'saved_models/')

  return config
