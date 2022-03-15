#!/usr/bin/env python
"""Generates energy efficient timing coef based on given speed."""
import os

from absl import app
from absl import flags

from ml_collections.config_flags import config_flags
import numpy as np
import rospy
import torch

from a1_interface.msg import gait_command
from perception.msg import image_embedding
from simple_gait_optimization.timing_policy.manual_gait_policy import ManualGaitPolicy

config_flags.DEFINE_config_file(
    'config', '/home/yxyang/research/semantic_locomotion_ros/src/'
    'simple_gait_optimization/src/simple_gait_optimization/'
    'configs/config_human.py')
flags.DEFINE_string(
    'model_dir', '/home/yxyang/research/semantic_locomotion_ros/src/'
    'simple_gait_optimization/src/simple_gait_optimization/saved_models/',
    'model_dir')

FLAGS = flags.FLAGS


def get_max_forward_speed(step_freq, max_swing_distance=0.3):
  return 2 * step_freq * max_swing_distance


def convert_to_linear_speed_equivalent(vx, vy, wz):
  return np.sqrt(vx**2 + 4 * vy**2) + np.abs(wz)


class GaitPolicy:
  """Records desired and actual speed from subscribed topics."""
  def __init__(self, model):
    self._model = model
    self._device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    self._last_embedding = None
    self._desired_speed = 0.
    self._current_speed = 0.
    self._policy = ManualGaitPolicy()

  def record_image_embedding(self, image_msg):
    self._last_embedding = np.array(image_msg.embedding)

  @property
  def last_embedding(self):
    return self._last_embedding

  def get_desired_speed_and_gait(self):
    """Returns desired speed and gait from policy."""
    if self._last_embedding is not None:
      model_inputs = torch.tensor(self._last_embedding).reshape(
          (1, -1)).to(self._device).float()
      pred_mean, pred_std = self._model.forward(model_inputs)
      pred_mean = pred_mean.detach().cpu().numpy()[0][0]
      pred_std = pred_std.detach().cpu().numpy()[0][0]
      desired_speed = pred_mean - 0.5 * pred_std
    else:
      desired_speed = 0.5
    gait = self._policy.get_action(desired_speed)
    gait.recommended_forward_speed = desired_speed
    return gait


def main(argv):
  del argv  # unused
  rospy.init_node('simple_gait_optimization', anonymous=True)

  # Load model and policy
  config = FLAGS.config
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
      'cpu')
  ckpt = torch.load(os.path.join(FLAGS.model_dir, 'trained_model.pkl'),
                    map_location=device)
  model = config.model_class(dim_in=config.model.dim_in,
                             dim_out=config.model.dim_out,
                             num_hidden=config.model.num_hidden,
                             dim_hidden=config.model.dim_hidden).to(device)
  model.load_state_dict(ckpt['model'])

  gait_policy = GaitPolicy(model)

  # Public to / subscribe from topics
  rospy.Subscriber('perception/image_embedding', image_embedding,
                   gait_policy.record_image_embedding)
  gait_command_publisher = rospy.Publisher('autogait_command',
                                           gait_command,
                                           queue_size=1)

  rate = rospy.Rate(20)
  while not rospy.is_shutdown():
    desired_gait = gait_policy.get_desired_speed_and_gait()
    gait_command_publisher.publish(desired_gait)
    rate.sleep()


if __name__ == "__main__":
  app.run(main)
