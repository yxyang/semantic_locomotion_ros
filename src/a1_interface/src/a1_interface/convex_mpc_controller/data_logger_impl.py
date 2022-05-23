"""Logs robot data and flushes periodically to the disk."""

from datetime import datetime
import os
import pickle
import rospy


class DataLoggerImpl:
  """Simple data logger implementation."""
  def __init__(self,
               logdir='/home/yxyang/research/semantic_locomotion_ros/logs',
               log_flush_seconds=2):
    self._buffer = []
    self._logdir = logdir
    self._log_flush_seconds = log_flush_seconds
    if not os.path.exists(self._logdir):
      os.makedirs(self._logdir)

  def update_logging(self, new_frame):
    self._buffer.append(new_frame)
    if (self._buffer[-1]['timestamp'] -
        self._buffer[0]['timestamp']) > self._log_flush_seconds:
      self.flush_logging()

  def flush_logging(self):
    rospy.loginfo("Start flushing...")
    logs = self._buffer
    self._buffer = []
    filename = 'log_{}.pkl'.format(
        datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    with open(os.path.join(self._logdir, filename), 'wb') as f:
      pickle.dump(logs, f)
    rospy.loginfo("Data logged to: {}".format(
        os.path.join(self._logdir, filename)))

  def close(self):
    self.flush_logging()
