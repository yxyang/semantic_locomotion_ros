"""Environment for evaluating gaits async under ROS integration."""

import numpy as np
import rospy

from a1_interface.convex_mpc_controller.gait_configs import slow
from a1_interface.msg import controller_mode, gait_type


class GaitChangeEnvAsync:
  """Environment for evaluating gaits async under ROS integration."""
  def __init__(self, dim_context=5, eval_duration=2, agent=None):
    self._agent = agent
    self._eval_duration = eval_duration

    self._last_episode_timestamp = rospy.get_rostime()
    slow_gait = slow.get_config()
    self._last_gait_command = gait_type(
        step_frequency=slow_gait.gait_parameters[0],
        foot_clearance=slow_gait.foot_clearance_max,
        base_height=slow_gait.desired_body_height,
        max_forward_speed=slow_gait.max_forward_speed,
        recommended_forward_speed=slow_gait.max_forward_speed,
        timestamp=rospy.get_rostime())
    self._last_image_embedding = np.zeros(dim_context)
    self._image_embedding_buffer = np.zeros(dim_context)
    self._speed_command_history = []

    self._gait_command_publisher = rospy.Publisher('gait_command',
                                                   gait_type,
                                                   queue_size=1)
    self._gait_command_publisher.publish(self._last_gait_command)

  def image_callback(self, msg):
    self._image_embedding_buffer = np.array(msg.embedding)

  def speed_command_callback(self, msg):
    """Receives speed command, computes reward and starts new episode."""
    self._speed_command_history.append(
        np.array((msg.vel_x, msg.vel_y, msg.rot_z)))
    if (self._last_episode_timestamp
        == 0) or (msg.timestamp.to_sec() -
                  self._last_episode_timestamp.to_sec() > self._eval_duration):
      # Record last episode
      avg_throttle_ratio = np.mean(self._speed_command_history, axis=0)[0]
      reward = self._last_gait_command.max_forward_speed * avg_throttle_ratio
      avg_turning = np.mean(self._speed_command_history, axis=0)[2]
      if avg_turning < 0.1:
        self._agent.receive_observation(self._last_image_embedding,
                                        self._last_gait_command,
                                        reward,
                                        refit_gp=False)
        print("Context: {}, Action: {}, Reward: {}".format(
            self._last_image_embedding, self._last_gait_command, reward))

      # Start a new episode
      self._last_episode_timestamp = msg.timestamp
      self._last_image_embedding = self._image_embedding_buffer.copy()
      self._last_gait_command = self._agent.get_suggestion(
          self._last_image_embedding)
      self._gait_command_publisher.publish(self._last_gait_command)
      self._speed_command_history = []

  def autogait_callback(self, msg):
    if msg.data != "GaitMode.AUTOGAIT_TRAIN":
      self._last_episode_timestamp = rospy.get_rostime()
      self._speed_command_history = []

  def controller_mode_callback(self, msg):
    if msg.mode != controller_mode.WALK:
      self._last_episode_timestamp = rospy.get_rostime()
      self._speed_command_history = []
