"""Implementation of GP-UCB algorithm for continuous bandits."""
import os
from threading import Lock
import warnings

from gym import spaces
import numpy as np
import rospy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from a1_interface.convex_mpc_controller.gait_configs import slow
from a1_interface.msg import gait_type


def get_max_forward_speed(step_freq, max_swing_distance=0.3):
  return 2 * step_freq * max_swing_distance


class CGPUCB:
  """The GP-UCB algorithm for continuous bandits."""
  def __init__(
      self,
      action_space: spaces.Box,
      kappa: float = 1.8,  #.7,
      num_samples: int = 10000,
      num_cem_iterations: int = 5,
      num_cem_elite_samples: int = 1000,
      dim_context=1):
    self.action_space = action_space
    self._dim_context = dim_context
    self._kappa = kappa
    self._num_samples = num_samples
    self._num_cem_iterations = num_cem_iterations
    self._num_cem_elite_samples = num_cem_elite_samples
    self.pipeline = self._construct_gp()
    self.is_fitted = False
    self._data_lock = Lock()

    self.context_history = np.zeros((0, self._dim_context))
    self.action_history = np.zeros((0, self.action_space.high.shape[0]))
    self.reward_history = np.zeros([0])
    self.reset()

  def _construct_gp(self):
    scaler = StandardScaler()
    gp = GaussianProcessRegressor(
        kernel=Matern(nu=2.5, length_scale_bounds=(1e-5, 1e5)) +
        WhiteKernel(noise_level=1, noise_level_bounds=(1e-5, 1e5)),
        n_restarts_optimizer=25,
        normalize_y=True,
        copy_X_train=True)
    return Pipeline([('scaler', scaler), ('gp', gp)])

  def get_suggestion(self, context) -> gait_type:
    """Gets action suggestion by maximizing acquisition function.

    The optimization for maximal acquisition function value is done via
    Cross Entropy Method (CEM)
    """
    if not self.is_fitted:
      slow_gait = slow.get_config()
      return gait_type(step_frequency=slow_gait.gait_parameters[0],
                       foot_clearance=slow_gait.foot_clearance_max,
                       base_height=slow_gait.desired_body_height,
                       max_forward_speed=slow_gait.max_forward_speed,
                       recommended_forward_speed=slow_gait.max_forward_speed,
                       timestamp=rospy.get_rostime())

    curr_mean = (self.action_space.high + self.action_space.low) / 2
    curr_std = (self.action_space.high - self.action_space.low) / 4
    for _ in range(self._num_cem_iterations):
      sampled_actions = np.random.normal(
          loc=curr_mean,
          scale=curr_std,
          size=[self._num_samples, self.action_space.low.shape[0]])
      sampled_actions = np.clip(sampled_actions, self.action_space.low,
                                self.action_space.high)
      sampled_contexts = np.stack([context] * self._num_samples, axis=0)
      sampled_inputs = np.concatenate((sampled_contexts, sampled_actions),
                                      axis=1)
      pred_mean, pred_std = self.pipeline.predict(sampled_inputs,
                                                  return_std=True)
      acquisition_function_values = pred_mean + self._kappa * pred_std
      best_action_indices = np.argsort(
          -acquisition_function_values)[:self._num_cem_elite_samples]
      elite_actions = sampled_actions[best_action_indices]
      elite_values = acquisition_function_values[best_action_indices]
      curr_mean = np.mean(elite_actions, axis=0)
      curr_std = np.std(elite_actions, axis=0)

    action = curr_mean
    max_speed = get_max_forward_speed(action[0])
    return gait_type(step_frequency=action[0],
                     foot_clearance=action[1],
                     base_height=action[2],
                     recommended_forward_speed=np.minimum(
                         max_speed, np.mean(elite_values)),
                     max_forward_speed=max_speed)

  def receive_observation(self,
                          context,
                          action: gait_type,
                          reward: float,
                          refit_gp: bool = True) -> None:
    """Receives observation and re-fits GP."""
    action_vec = np.array(
        (action.step_frequency, action.foot_clearance, action.base_height))
    with self._data_lock:
      self.context_history = np.concatenate((self.context_history, [context]),
                                            axis=0)
      self.action_history = np.concatenate((self.action_history, [action_vec]),
                                           axis=0)
      self.reward_history = np.concatenate((self.reward_history, [reward]),
                                           axis=0)

    if refit_gp:
      self.refit_gp()

  def refit_gp(self) -> None:
    """Refits GP based on latest observations."""
    if self.context_history.shape[0] <= 0:
      return

    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      new_pipeline = self._construct_gp()
      new_pipeline.fit(
          np.concatenate((self.context_history, self.action_history), axis=1),
          self.reward_history)
    self.is_fitted = True
    self.pipeline = new_pipeline

  def reset(self) -> None:
    with self._data_lock:
      self.context_history = np.zeros((0, self._dim_context))
      self.action_history = np.zeros((0, self.action_space.high.shape[0]))
      self.reward_history = np.zeros([0])

  def save(self, logdir: str) -> None:
    """Saves historical data to disk."""
    if not os.path.exists(logdir):
      os.makedirs(logdir)

    filename = os.path.join(logdir, 'checkpoint.npz')
    with open(filename, "wb") as f:
      np.savez(f,
               context_history=self.context_history,
               action_history=self.action_history,
               reward_history=self.reward_history)
    rospy.loginfo("Saved checkpoint to: {}.".format(filename))

  def restore(self, logdir: str) -> None:
    """Restores checkpoint from disk."""
    filename = os.path.join(logdir, 'checkpoint.npz')
    ckpt = dict(np.load(open(filename, 'rb')))
    self.context_history = ckpt['context_history']
    self.action_history = ckpt['action_history']
    self.reward_history = ckpt['reward_history']
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      self.pipeline.fit(
          np.concatenate((self.context_history, self.action_history), axis=1),
          self.reward_history)
    rospy.loginfo("Restored from: {}".format(filename))

  @property
  def iter_count(self):
    return self.reward_history.shape[0]
