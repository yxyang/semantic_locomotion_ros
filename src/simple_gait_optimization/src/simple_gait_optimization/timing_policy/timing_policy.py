#!/usr/bin/env python
"""Generates gait timing from desired and actual speed."""
import os

from absl import app

import numpy as np
import rospkg

ACTION_HIGH = np.array([4, 1, 1, 1, 1, 1, 1, 0.99])
ACTION_LOW = np.array([0.001, -1, -1, -1, -1, -1, -1, 0.01])


def relu(x):
  return x * (x > 0)


class TimingPolicy:
  """Outputs timing commands based on desired and actual speed of the robot."""
  def __init__(self):
    package_path = rospkg.RosPack().get_path('simple_gait_optimization')
    ckpt_dir = os.path.join(package_path, 'data', 'refined_weights.npz')
    weights = np.load(open(ckpt_dir, 'rb'))
    self.w1, self.b1 = weights['w1'], weights['b1']
    self.w2, self.b2 = weights['w2'], weights['b2']

  def get_action(self, base_vel, desired_vel):
    """Performs network inference."""
    obs = np.concatenate((base_vel, desired_vel))
    hidden = relu(self.w1.dot(obs) + self.b1)
    output = self.w2.dot(hidden) + self.b2
    output = np.tanh(output)

    action_mid = (ACTION_LOW + ACTION_HIGH) / 2
    action_range = (ACTION_HIGH - ACTION_LOW) / 2
    action = output * action_range + action_mid

    return np.concatenate((action[:1], np.arctan2(action[1:4],
                                                  action[4:7]), action[-1:]))


def main(_):
  policy = TimingPolicy()
  base_vel = np.zeros(3)
  desired_vel = np.zeros(4)
  print(policy.get_action(base_vel, desired_vel))


if __name__ == "__main__":
  app.run(main)
