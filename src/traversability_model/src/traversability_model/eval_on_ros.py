#!/usr/bin/env python
"""Evaluates the trained traversability model on a set of images."""
import os

from absl import app
from absl import flags

from ml_collections.config_flags import config_flags
import numpy as np
import rospy
import torch

from a1_interface.msg import gait_type
from perception.msg import image_embedding

config_flags.DEFINE_config_file(
    'config',
    ('/home/yxyang/research/semantic_locomotion_ros/src/traversability_model/'
     'src/traversability_model/configs/config_human.py'))
flags.DEFINE_string(
    'model_dir',
    '/home/yxyang/research/semantic_locomotion_ros/src/traversability_model/'
    'src/traversability_model/saved_models/', 'model_dir')
FLAGS = flags.FLAGS


class GaitPolicy:
  """Gait policy based on human labels."""
  def __init__(self, models):
    self.models = models
    self._device = torch.device('cpu')
    self._last_embedding = None

  def callback(self, msg):
    """Receives embedding from perception module."""
    self._last_embedding = np.array(msg.embedding)

  @property
  def last_embedding(self):
    return self._last_embedding

  def get_desired_gait(self):
    """Returns desired gait based on model predictions."""
    model_inputs = torch.tensor(self._last_embedding).reshape(
        (1, -1)).to(self._device).float()
    pred_means, pred_stds = [], []
    for model in self.models:
      pred_mean, pred_std = model.forward(model_inputs)
      pred_mean = pred_mean.cpu().detach().numpy()[:, 0]
      pred_std = pred_std.cpu().detach().numpy()[:, 0]
      pred_means.append(pred_mean[0])
      pred_stds.append(pred_std[0])

    pred_means = np.array(pred_means)
    pred_stds = np.array(pred_stds)
    lcb = pred_means - pred_stds + np.array([0.4, 0.2, 0])
    desired_gait = np.argmax(lcb)
    return gait_type(timestamp=rospy.get_rostime(), type=desired_gait)


def main(argv):
  del argv  # unused
  rospy.init_node("gait_policy", anonymous=True)
  config = FLAGS.config
  # Restore model inputs
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
      'cpu')
  ckpt = torch.load(os.path.join(FLAGS.model_dir, 'trained_model.pkl'), map_location=torch.device('cpu'))
  gait_names = ['crawl', 'walk', 'run']
  gait_models = []
  for gait_name in gait_names:
    model = config.model_class(dim_in=config.model.dim_in,
                               dim_out=config.model.dim_out,
                               num_hidden=config.model.num_hidden,
                               dim_hidden=config.model.dim_hidden).to(device)
    model.load_state_dict(ckpt['model_{}'.format(gait_name)])
    gait_models.append(model)

  policy = GaitPolicy(gait_models)
  rospy.Subscriber('perception/image_embedding', image_embedding,
                   policy.callback)
  gait_command_publisher = rospy.Publisher('gait_command',
                                           gait_type,
                                           queue_size=1)
  rate = rospy.Rate(6)
  while not rospy.is_shutdown():
    if policy.last_embedding is not None:
      gait_command = policy.get_desired_gait()
      gait_command_publisher.publish(gait_command)
    rate.sleep()


if __name__ == "__main__":
  app.run(main)
