#!/usr/bin/env python
"""Selects gaits based on traversability score."""
import collections

import numpy as np
import rospy

from a1_interface.msg import gait_type

class FixedRegionGaitPolicy:
  """Decides gait based on traversability score."""
  def __init__(self):
    self._traversability_score_history = collections.deque(maxlen=10)
    self._traversability_score_history.append(1.2)

  def update_score(self, new_score):
    self._traversability_score_history.append(new_score.data)

  def get_gait_action(self):
    avg_traversability_score = np.mean(self._traversability_score_history)
    if avg_traversability_score < 0.5:
      return gait_type(type=gait_type.SLOW, timestamp=rospy.get_rostime())
    elif avg_traversability_score < 1.5:
      return gait_type(type=gait_type.SLOW, timestamp=rospy.get_rostime())
    elif avg_traversability_score < 2.5:
      return gait_type(type=gait_type.MID, timestamp=rospy.get_rostime())
    else:
      return gait_type(type=gait_type.FAST, timestamp=rospy.get_rostime())
