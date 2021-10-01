"""Build a simple world with plane only."""


class BlocksWorld:
  """Builds a simple world with a plane only."""
  def __init__(self, pybullet_client):
    self._pybullet_client = pybullet_client

  def build_world(self):
    ground_id = self._pybullet_client.loadURDF('blocks.urdf')
    self._pybullet_client.changeDynamics(ground_id, -1, lateralFriction=1.)
    return ground_id
