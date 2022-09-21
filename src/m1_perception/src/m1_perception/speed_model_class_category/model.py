"""Simple FC Network to predict speed from semantic embedding."""
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


class SpeedModel(Model):
  """A model that estimates speed from semantic embedding."""
  def __init__(self, num_hidden_layers=1, dim_hidden=20, **kwargs):
    super().__init__(**kwargs)
    self._layers = []
    for _ in range(num_hidden_layers):
      self._layers.append(Dense(dim_hidden, activation='relu'))
    self._layers.append(Dense(1, activation=None))

  def call(self, x):
    # Perform PCA to reduce dimensions
    for layer in self._layers:
      x = layer(x)
    return x[..., 0]
