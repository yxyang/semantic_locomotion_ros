"""Implementation of GP-UCB algorithm for continuous bandits."""
import os
from typing import Sequence
import warnings

from gym import spaces
import numpy as np
import rospy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class GPUCB:
  """The GP-UCB algorithm for continuous bandits."""
  def __init__(
      self,
      action_space: spaces.Box,
      kappa: float = 1.8,  #.7,
      num_samples: int = 10000,
      num_cem_iterations: int = 5,
      num_cem_elite_samples: int = 1000):
    self.action_space = action_space
    self._kappa = kappa
    self._num_samples = num_samples
    self._num_cem_iterations = num_cem_iterations
    self._num_cem_elite_samples = num_cem_elite_samples
    self.scaler = StandardScaler()
    self.gp = GaussianProcessRegressor(
        kernel=Matern(nu=2.5, length_scale_bounds=(0.01, 1e5)) +
        WhiteKernel(noise_level_bounds=(0.1, 0.5)),
        n_restarts_optimizer=25,
        normalize_y=True)
    self.pipeline = Pipeline([('scaler', self.scaler), ('gp', self.gp)])

    self.action_history = np.zeros((0, self.action_space.high.shape[0]))
    self.reward_history = np.zeros([0])
    self.reset()

  def get_suggestion(self) -> Sequence[float]:
    """Gets action suggestion by maximizing acquisition function.

    The optimization for maximal acquisition function value is done via
    Cross Entropy Method (CEM)
    """
    if len(self.action_history) == 0:
      return self.action_space.sample()

    curr_mean = (self.action_space.high + self.action_space.low) / 2
    curr_std = (self.action_space.high - self.action_space.low) / 4
    for _ in range(self._num_cem_iterations):
      sampled_actions = np.random.normal(
          loc=curr_mean,
          scale=curr_std,
          size=[self._num_samples, self.action_space.low.shape[0]])
      sampled_actions = np.clip(sampled_actions, self.action_space.low,
                                self.action_space.high)
      pred_mean, pred_std = self.pipeline.predict(sampled_actions,
                                                  return_std=True)
      acquisition_function_values = pred_mean + self._kappa * pred_std
      best_action_indices = np.argsort(
          -acquisition_function_values)[:self._num_cem_elite_samples]
      elite_actions = sampled_actions[best_action_indices]
      curr_mean = np.mean(elite_actions, axis=0)
      curr_std = np.std(elite_actions, axis=0)

    return curr_mean

  def receive_observation(self, action: Sequence[float],
                          reward: float) -> None:
    self.action_history = np.concatenate((self.action_history, [action]),
                                         axis=0)
    self.reward_history = np.concatenate((self.reward_history, [reward]),
                                         axis=0)
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      self.pipeline.fit(self.action_history, self.reward_history)

  def reset(self) -> None:
    self.action_history = np.zeros((0, self.action_space.high.shape[0]))
    self.reward_history = np.zeros([0])

  def save(self, logdir: str) -> None:
    if not os.path.exists(logdir):
      os.makedirs(logdir)

    filename = os.path.join(logdir, 'checkpoint.npz')
    with open(filename, "wb") as f:
      np.savez(f,
               action_history=self.action_history,
               reward_history=self.reward_history)
    rospy.loginfo("Saved checkpoint to: {}.".format(filename))

  def restore(self, logdir: str) -> None:
    filename = os.path.join(logdir, 'checkpoint.npz')
    ckpt = dict(np.load(open(filename, 'rb')))
    self.action_history = ckpt['action_history']
    self.reward_history = ckpt['reward_history']
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      self.pipeline.fit(self.action_history, self.reward_history)
    rospy.loginfo("Restored from: {}".format(filename))
