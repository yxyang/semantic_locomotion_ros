#!/usr/bin/env python
"""Optimize gaits using GP in a single environment."""
from absl import app
from absl import flags

from gym import spaces
import numpy as np

from a1_interface.worlds import plane_world, uneven_world, soft_world, slippery_world, slope_world
from gait_optimizer.bayesian_optimization import gp_ucb
from gait_optimizer.envs import fixed_env

flags.DEFINE_string('logdir', None, 'where to log experiments.')
flags.DEFINE_integer('num_iter', 10, 'Number of iterations to run.')
flags.DEFINE_string('restore_checkpoint', None,
                    'whether to restore previous checkpoint.')
flags.DEFINE_enum('world_name', 'plane',
                  ['plane', 'soft', 'uneven', 'slippery', 'slope', 'real'],
                  'which world to run.')
flags.DEFINE_bool('show_gui', False, 'whether to show gui.')
FLAGS = flags.FLAGS

PARAM_LB = np.array([1., 0.08, 0.24, 0.1])
PARAM_UB = np.array([3.5, 0.18, 0.3, 2.0])
WORLD_NAME_TO_WORLD_CLASS = {
    'plane': plane_world.PlaneWorld,
    'soft': soft_world.SoftWorld,
    'uneven': uneven_world.UnevenWorld,
    'slippery': slippery_world.SlipperyWorld,
    'slope': slope_world.SlopeWorld,
}


def main(_):
  action_space = spaces.Box(low=PARAM_LB, high=PARAM_UB)
  agent = gp_ucb.GPUCB(action_space)
  if FLAGS.restore_checkpoint:
    agent.restore(FLAGS.restore_checkpoint)

  if FLAGS.world_name == 'real':
    env = fixed_env.FixedEnv(plane_world.PlaneWorld,
                             show_gui=False,
                             use_real_robot=True)
  else:
    world_class = WORLD_NAME_TO_WORLD_CLASS[FLAGS.world_name]
    env = fixed_env.FixedEnv(world_class,
                             show_gui=FLAGS.show_gui,
                             use_real_robot=False)

  for i in range(FLAGS.num_iter):
    action = agent.get_suggestion()
    reward = env.eval_parameters(action)
    agent.receive_observation(action, reward)
    print("Iter: {}, action: {}, reward: {}".format(i, action, reward))
    if FLAGS.logdir:
      agent.save(FLAGS.logdir)

  env.close()

if __name__ == '__main__':
  app.run(main)
