#!/usr/bin/env python
"""Selects gaits based on traversability score."""
from absl import app
from absl import flags
import collections
import numpy as np
import rospy
from std_msgs.msg import Float32

from a1_interface.msg import gait_type

FLAGS = flags.FLAGS

class FixedRegionGaitPolicy:
  """Decides gait based on traversability score."""
  def __init__(self):
    self._traversability_score_history = collections.deque(maxlen=10)
    self._traversability_score_history.append(1.2)

  def update_score(self, new_score):
    self._traversability_score_history.append(new_score.data)

  def get_gait_action(self):
    avg_traversability_score = np.mean(self._traversability_score_history)
    if avg_traversability_score < 0.75:
      return gait_type(type=gait_type.TROT, timestamp=rospy.get_rostime())
    else:
      return gait_type(type=gait_type.CRAWL, timestamp=rospy.get_rostime())


def main(_):
  policy = FixedRegionGaitPolicy()
  rospy.Subscriber("/perception/traversability_score", Float32,
                   policy.update_score)
  gait_type_publisher = rospy.Publisher('gait_type', gait_type, queue_size=1)
  rospy.init_node('fixed_region_gait_policy', anonymous=True)

  rate = rospy.Rate(5)
  while not rospy.is_shutdown():
    gait_type_publisher.publish(policy.get_gait_action())
    rate.sleep()

if __name__ == "__main__":
  app.run(main)
