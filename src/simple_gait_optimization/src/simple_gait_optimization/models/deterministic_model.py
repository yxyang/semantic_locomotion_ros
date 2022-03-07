"""Implements a deterministic model to predict terrain traversability."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import trange

from simple_gait_optimization.models.utils import swish, get_affine_params


class DeterministicModel(nn.Module):
  """Deterministic Neural Network model for regression."""
  def __init__(self, dim_in, dim_out, num_hidden=1, dim_hidden=50):
    super().__init__()
    self._dim_in = dim_in
    self._dim_out = dim_out
    self._dim_hidden = dim_hidden
    self._num_hidden = num_hidden

    # Construct nn weights
    self.weights, self.biases = [], []
    # Input layer
    w, b = get_affine_params(self._dim_in, self._dim_hidden)
    self.weights.append(w)
    self.biases.append(b)

    # Hidden layers:
    for _ in range(1, self._num_hidden):
      w, b = get_affine_params(self._dim_hidden, self._dim_hidden)
      self.weights.append(w)
      self.biases.append(b)

    w, b = get_affine_params(self._dim_hidden, self._dim_out)
    self.weights.append(w)
    self.biases.append(b)
    self.weights = torch.nn.ParameterList(self.weights)
    self.biases = torch.nn.ParameterList(self.biases)

    self.inputs_mean = nn.Parameter(torch.zeros(self._dim_in),
                                    requires_grad=False)
    self.inputs_std = nn.Parameter(torch.ones(self._dim_in),
                                   requires_grad=False)

  def fit_input_stats(self, inputs, device):
    mean = np.mean(inputs, axis=0)
    std = np.std(inputs, axis=0)
    std[std < 1e-12] = 1.

    self.inputs_mean.data = torch.from_numpy(mean).float().to(device)
    self.inputs_std.data = torch.from_numpy(std).float().to(device)

  def forward(self, inputs, return_logstd=False):
    del return_logstd  # unused
    curr = (inputs - self.inputs_mean) / self.inputs_std

    for (weight, bias) in zip(self.weights[:-1], self.biases[:-1]):
      curr = curr.matmul(weight) + bias
      curr = swish(curr)
    curr = curr.matmul(self.weights[-1]) + self.biases[-1]

    return curr, torch.zeros_like(curr)

  def compute_loss(self, inputs, labels):
    pred_mean, _ = self.forward(inputs, return_logstd=True)
    loss = (pred_mean - labels)**2
    loss = loss.mean()
    return loss

  def train(self, train_set, val_set, num_epochs=1, batch_size=100):
    """Train the model using SGD."""
    optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=False)

    epoch_range = trange(num_epochs, unit="epoch(s)", desc="Training")
    for _ in epoch_range:
      train_losses = []
      for inputs, labels in train_loader:
        loss = self.compute_loss(inputs, labels)
        train_losses.append(loss.cpu().detach().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

      # Validation
      val_losses = []
      for inputs, labels in val_loader:
        val_losses.append(
            self.compute_loss(inputs, labels).cpu().detach().numpy())

      epoch_range.set_postfix({
          "Train Loss": np.mean(train_losses),
          "Val Loss": np.mean(val_losses)
      })
