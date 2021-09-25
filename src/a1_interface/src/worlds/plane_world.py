"""Build a simple world with plane only."""


class PlaneWorld:
  """Builds a simple world with a plane only."""
  def __init__(self, pybullet_client):
    self._pybullet_client = pybullet_client

  def build_world(self):
    return self._pybullet_client.loadURDF('plane.urdf')
