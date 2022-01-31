#!/usr/bin/env python
"""Evaluates the trained traversability model on a set of images."""
import multiprocessing
import os

from absl import app
from absl import flags

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ml_collections.config_flags import config_flags
import numpy as np
from sklearn.decomposition import PCA
import torch
from tqdm import tqdm

config_flags.DEFINE_config_file(
    'config',
    'src/traversability_model/src/traversability_model/configs/config_human.py'
)
flags.DEFINE_string(
    'model_dir',
    'src/traversability_model/src/traversability_model/saved_models/',
    'model_dir')
flags.DEFINE_string('image_embeddings_dir',
                    ('src/traversability_model/src/traversability_model/data/'
                     'vision_embeddings.npz'),
                    'path to vision embeddings file.')
flags.DEFINE_string('image_dir', 'logs/full_outdoor_tx2/ghost_memory',
                    'where image files are stored.')
flags.DEFINE_string('output_dir', 'logs/human_ratio', 'output_dir.')
FLAGS = flags.FLAGS


def generate_plot_and_save(args):
  """Generates model visualization plot and save to disk."""
  _, base_dir, output_dir, filename, pred_means, pred_stds = args
  pred_means = (pred_means + np.array([0.4, 0.2, 0])) * 0.34 + 0.65
  pred_stds = pred_stds * 0.34
  fig = plt.figure(figsize=(8 * 1.5, 2.25 * 1.5))
  plt.subplot(1, 2, 1)
  plt.axis('off')
  plt.imshow(mpimg.imread(os.path.join(base_dir, filename)))
  plt.subplot(1, 2, 2)
  # plt.gca().get_xaxis().set_visible(False)

  lcb = pred_means - pred_stds
  best_gait = np.argmax(lcb)
  if best_gait == 0:
    title = "Best: Crawl"
  elif best_gait == 1:
    title = "Best: Walk"
  else:
    title = "Best: Run"

  plt.title(title, fontsize=20)
  gait_names = ['Crawl', 'Walk', 'Run']

  for idx in range(3):
    color = '#5273ab' if idx != best_gait else '#e32f27'
    plt.errorbar([gait_names[idx]],
                 pred_means[idx],
                 yerr=pred_stds[idx],
                 capsize=10,
                 capthick=5,
                 elinewidth=5,
                 markersize=10,
                 fmt='o',
                 color=color)
  plt.xticks(fontsize=20)
  plt.ylim(0, 1.6)
  plt.xlim(-1, 3)
  plt.ylabel('Predicted Speed (m/s)', fontsize=15)

  plt.savefig(os.path.join(output_dir, filename), format='png')
  # plt.savefig(os.path.join(output_dir, "image_{:05d}.png".format(idx)),
  # format='png')
  plt.close(fig)
  return best_gait, lcb[best_gait]


def moving_average(data, window_size=60):
  ans = np.zeros_like(data)
  for idx in range(data.shape[0]):
    start_idx = np.maximum(0, idx - window_size + 1)
    ans[idx] = np.mean(data[start_idx:idx + 1], axis=0)
  return ans


def main(argv):
  del argv  # unused
  matplotlib.use('Agg')
  plt.ioff()

  config = FLAGS.config
  # Restore model inputs
  vision_data = dict(
      np.load(open(FLAGS.image_embeddings_dir, 'rb'), allow_pickle=True))
  vision_inputs = moving_average(
      vision_data['embeddings'],
      window_size=config.feature_moving_average_window_size)
  pca_data = np.load(
      open(os.path.join(FLAGS.model_dir, 'pca_training_data.npz'), 'rb'))
  pca = PCA(n_components=config.pca_output_dim)
  pca.fit(pca_data['pca_train_data'])
  vision_inputs = pca.transform(vision_inputs)

  # Restore models and generate model predictions
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
      'cpu')
  ckpt = torch.load(os.path.join(FLAGS.model_dir, 'trained_model.pkl'))
  model = config.model_class(dim_in=config.model.dim_in,
                             dim_out=config.model.dim_out,
                             num_hidden=config.model.num_hidden,
                             dim_hidden=config.model.dim_hidden).to(device)
  pred_means, pred_stds = [], []

  gait_names = ['crawl', 'walk', 'run']
  for gait_name in gait_names:
    model.load_state_dict(ckpt['model_{}'.format(gait_name)])
    model_inputs = torch.tensor(vision_inputs).to(device).float()
    pred_mean, pred_std = model.forward(model_inputs)
    pred_mean = pred_mean.cpu().detach().numpy()[:, 0]
    pred_std = pred_std.cpu().detach().numpy()[:, 0]
    pred_means.append(pred_mean)
    pred_stds.append(pred_std)

  filelist = sorted(os.listdir(FLAGS.image_dir))
  args = []
  for idx in range(vision_inputs.shape[0]):
    means = np.array([pred_mean[idx] for pred_mean in pred_means])
    stds = np.array([pred_std[idx] for pred_std in pred_stds])
    curr_arg = [
        idx, FLAGS.image_dir, FLAGS.output_dir, filelist[idx], means, stds
    ]
    args.append(curr_arg)

  with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as p:
    ans = list(tqdm(p.imap(generate_plot_and_save, args), total=len(args)))

  gait_choices = np.array([result[0] for result in ans])
  speed_choices = np.array([result[1] for result in ans])
  np.savez(os.path.join(FLAGS.output_dir, 'commands.npz'),
           gait_choices=gait_choices,
           speed_choices=speed_choices,
           **vision_data)


if __name__ == "__main__":
  app.run(main)
