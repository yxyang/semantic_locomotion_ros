#!/usr/bin/env python
"""Example of running the GP-UCB algorithm"""
from absl import app
from absl import flags

from gym import spaces
import numpy as np

from gait_optimizer.bayesian_optimization import gp_ucb

flags.DEFINE_integer('num_iter', 10, "Number of iterations to run gp-ucb for.")
FLAGS = flags.FLAGS


def target_function(x: float) -> float:
  return np.sin(3 * x) + 0.1 * x


def main(_):
  action_space = spaces.Box(high=np.array([3]), low=np.array([-3]))
  agent = gp_ucb.GPUCB(action_space)
  for i in range(FLAGS.num_iter):
    action = agent.get_suggestion()
    reward = target_function(action[0])
    agent.receive_observation(action, reward)
    print("Iter: {}, action: {}, reward: {}".format(i, action, reward))


if __name__ == "__main__":
  app.run(main)
