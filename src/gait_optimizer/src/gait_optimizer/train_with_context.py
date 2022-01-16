#!/usr/bin/env python
"""Optimize gaits using GP in a single environment."""
import os

from absl import app
from absl import flags

from gym import spaces
import numpy as np

from a1_interface.worlds import plane_world, random_world
from gait_optimizer.bayesian_optimization import cgp_ucb
from gait_optimizer.envs import gait_change_env

flags.DEFINE_string('logdir', None, 'where to log experiments.')
flags.DEFINE_integer('num_iter', 10, 'Number of iterations to run.')
flags.DEFINE_string('restore_checkpoint', None,
                    'whether to restore previous checkpoint.')
flags.DEFINE_enum('world_name', 'sim', ['sim', 'real'], 'which world to run.')
flags.DEFINE_bool('show_gui', False, 'whether to show gui.')
flags.DEFINE_bool('use_real_camera', False,
                  'whether to use physical camera or synthetic camera.')
FLAGS = flags.FLAGS

PARAM_LB = np.array([1.5, 0.08, 0.24, 0.1])
PARAM_UB = np.array([3.5, 0.18, 0.3, 1.0])
np.set_printoptions(suppress=True, precision=2)


def main(_):
  action_space = spaces.Box(low=PARAM_LB, high=PARAM_UB)
  agent = cgp_ucb.CGPUCB(action_space, dim_context=4)
  if FLAGS.restore_checkpoint:
    agent.restore(FLAGS.restore_checkpoint)

  if FLAGS.world_name == 'real':
    env = gait_change_env.GaitChangeEnv(plane_world.PlaneWorld,
                                        show_gui=False,
                                        use_real_robot=True,
                                        use_real_camera=True)
  else:
    env = gait_change_env.GaitChangeEnv(random_world.RandomWorld,
                                        show_gui=FLAGS.show_gui,
                                        use_real_robot=False,
                                        use_real_camera=FLAGS.use_real_camera)

  for i in range(agent.iter_count, agent.iter_count + FLAGS.num_iter):
    env.reset()
    context = env.get_context()
    action = agent.get_suggestion(context)
    reward = env.eval_parameters(action)
    agent.receive_observation(context, action, reward)
    print("Iter: {}, context: {}, action: {}, reward: {}".format(
        i, context, action, reward))
    if FLAGS.logdir:
      agent.save(FLAGS.logdir)
      env.save_latest_trajectory(
          os.path.join(FLAGS.logdir, "traj_{}.pkl".format(i)))

  env.close()


if __name__ == '__main__':
  app.run(main)
