"""Build a simple world with plane only."""


class SoftWorld:
  """Builds a simple world with a plane only."""
  def __init__(self, pybullet_client):
    self._pybullet_client = pybullet_client

  def build_world(self):
    """Builds world with a simple plane and custom friction."""
    ground_id = self._pybullet_client.loadURDF('plane_red.urdf')
    self._pybullet_client.changeDynamics(ground_id,
                                         -1,
                                         contactDamping=10,
                                         contactStiffness=400)
    return ground_id
