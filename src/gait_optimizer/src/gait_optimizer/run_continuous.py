#!/usr/bin/env python
"""Optimize gaits using GP in a single environment."""
from absl import app
from absl import flags

from gym import spaces
import numpy as np
import rospy
from std_msgs.msg import String

from a1_interface.msg import controller_mode, speed_command
from gait_optimizer.bayesian_optimization import cgp_ucb
from gait_optimizer.envs import gait_change_env_async
from perception.msg import image_embedding

flags.DEFINE_string('logdir', 'logs', 'where to log experiments.')
flags.DEFINE_string('restore_checkpoint', None,
                    'whether to restore previous checkpoint.')
# TODO: group all following flags into a config file.
flags.DEFINE_integer('dim_context', 5, 'context dimension.')
flags.DEFINE_integer('eval_duration', 2, 'duration of each gait evaluation.')
FLAGS = flags.FLAGS

PARAM_LB = np.array([1.5, 0.08, 0.24])
PARAM_UB = np.array([3.5, 0.18, 0.3])


def main(_):
  rospy.init_node("gait_optimizer", anonymous=True)

  action_space = spaces.Box(low=PARAM_LB, high=PARAM_UB)
  agent = cgp_ucb.CGPUCB(action_space, dim_context=5)
  if FLAGS.restore_checkpoint:
    agent.restore(FLAGS.restore_checkpoint)

  env = gait_change_env_async.GaitChangeEnvAsync(
      dim_context=FLAGS.dim_context,
      eval_duration=FLAGS.eval_duration,
      agent=agent)

  rospy.Subscriber('perception/image_embedding', image_embedding,
                   env.image_callback)
  rospy.Subscriber('speed_command', speed_command, env.speed_command_callback)
  rospy.Subscriber('autogait', String, env.autogait_callback)
  rospy.Subscriber('controller_mode', controller_mode,
                   env.controller_mode_callback)

  rate = rospy.Rate(1. / FLAGS.eval_duration)
  while not rospy.is_shutdown():
    agent.refit_gp()
    agent.save(FLAGS.logdir)
    rate.sleep()


if __name__ == '__main__':
  app.run(main)
