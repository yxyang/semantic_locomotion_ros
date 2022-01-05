"""Implementation of GP-UCB algorithm for continuous bandits."""
import os
from typing import Sequence
import warnings

from gym import spaces
import numpy as np
import rospy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


class GPUCB:
  """The GP-UCB algorithm for continuous bandits."""
  def __init__(self,
               action_space: spaces.Box,
               kappa: float = .7,
               num_samples: int = 10000):
    self.action_space = action_space
    self._kappa = kappa
    self._num_samples = num_samples
    self.gp = GaussianProcessRegressor(kernel=Matern(nu=2.5),
                                       n_restarts_optimizer=25,
                                       normalize_y=True)
    self.action_history = []
    self.reward_history = []
    self.reset()

  def get_suggestion(self) -> Sequence[float]:
    """Gets action suggestion by maximizing acquisition function."""
    if not self.action_history:
      return self.action_space.sample()
    sampled_actions = np.random.uniform(
        self.action_space.low,
        self.action_space.high,
        size=[self._num_samples, self.action_space.low.shape[0]])
    pred_mean, pred_std = self.gp.predict(sampled_actions, return_std=True)
    acquisition_function_values = pred_mean + self._kappa * pred_std
    best_action_index = np.argmax(acquisition_function_values)
    return sampled_actions[best_action_index]

  def receive_observation(self, action: Sequence[float],
                          reward: float) -> None:
    self.action_history.append(action)
    self.reward_history.append(reward)
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      self.gp.fit(self.action_history, self.reward_history)

  def reset(self) -> None:
    self.action_history = []
    self.reward_history = []

  def save(self, logdir: str) -> None:
    if not os.path.exists(logdir):
      os.makedirs(logdir)

    filename = os.path.join(logdir, 'checkpoint.npz')
    with open(filename, "wb") as f:
      np.savez(f,
               action_history=self.action_history,
               reward_history=self.reward_history)
    rospy.loginfo("Saved checkpoint to: {}.".format(filename))

  def load(self, logdir: str) -> None:
    filename = os.path.join(logdir, 'checkpoint.npz')
    ckpt = dict(np.load(open(filename, 'wb')))
    self.action_history = ckpt['action_history']
    self.reward_history = ckpt['reward_history']
    rospy.loginfo("Restored from: {}".format(filename))
