"""FCHarDNet implementation in tensorflow"""
# from absl import logging

# import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, ReLU, AveragePooling2D


class TransitionUp(Layer):
  """Transition up with bilinear interpolation."""
  def call(self, inputs, skip=None, concat=True):
    # out = UpSampling2D(interpolation='bilinear')(inputs)
    out = tf.image.resize(inputs, skip.shape[1:3])

    if concat:
      skip = tf.dtypes.cast(skip, tf.float32)
      out = tf.concat([out, skip], 3)

    return out


class ConvLayer(Layer):
  """Convolution layer with BatchNorm and ReLU activation."""
  def __init__(self, out_channels, kernel=3, stride=1, **kwargs):
    super().__init__(**kwargs)

    self.conv = Conv2D(out_channels,
                       kernel_size=kernel,
                       strides=stride,
                       padding='same',
                       use_bias=False)
    self.norm = BatchNormalization()
    self.relu = ReLU()

  def call(self, inputs):
    out = self.conv(inputs)
    out = self.norm(out)
    out = self.relu(out)
    return out


class HardBlock(Layer):
  """One block of HarDNet"""
  def get_link(self, layer, base_ch, growth_rate, grmul):
    """Returns desired link between layers."""
    if layer == 0:
      return base_ch, 0, []
    out_channels = growth_rate
    link = []
    for i in range(10):
      dv = 2**i
      if layer % dv == 0:
        k = layer - dv
        link.append(k)
        if i > 0:
          out_channels *= grmul
    out_channels = int(int(out_channels + 1) / 2) * 2
    in_channels = 0
    for i in link:
      ch, _, _ = self.get_link(i, base_ch, growth_rate, grmul)
      in_channels += ch
    return out_channels, in_channels, link

  def __init__(self,
               in_channels,
               growth_rate,
               grmul,
               n_layers,
               keep_base=False,
               **kwargs):
    super().__init__(**kwargs)
    self.keep_base = keep_base
    self.links = []
    self.out_channels = 0
    self.layers = []

    for i in range(n_layers):
      outch, _, link = self.get_link(i + 1, in_channels, growth_rate, grmul)
      self.links.append(link)
      self.layers.append(ConvLayer(outch))
      if (i % 2 == 0) or (i == n_layers - 1):
        self.out_channels += outch

  def get_out_ch(self):
    return self.out_channels

  def call(self, inputs):
    """Forward pass of HardBlock."""
    layers_ = [inputs]

    for layer in range(len(self.layers)):
      link = self.links[layer]
      tin = []
      for i in link:
        tin.append(layers_[i])
      if len(tin) > 1:
        inputs = tf.concat(tin, 3)
      else:
        inputs = tin[0]
      out = self.layers[layer](inputs)
      layers_.append(out)
    t = len(layers_)
    out_ = []
    for i in range(t):
      if (i == 0 and self.keep_base) or (i == t - 1) or (i % 2 == 1):
        out_.append(layers_[i])
    out = tf.concat(out_, 3)
    return out


class HardNet(Model):
  """The complete HarDNet Model."""
  def __init__(self, n_classes=1, **kwargs):
    super().__init__(**kwargs)
    self._pca_mean, self._principal_components = None, None

    first_ch = [16, 24, 32, 48]
    ch_list = [64, 96, 160, 224, 320]
    grmul = 1.7
    gr = [10, 16, 18, 24, 32]
    n_layers = [4, 4, 8, 8, 8]

    blks = len(n_layers)

    self.shortcut_layers = []

    self.base = []
    self.base.append(ConvLayer(out_channels=first_ch[0], kernel=3, stride=2))
    self.base.append(ConvLayer(first_ch[1], kernel=3))
    self.base.append(ConvLayer(first_ch[2], kernel=3, stride=2))
    self.base.append(ConvLayer(first_ch[3], kernel=3))

    skip_connection_channel_counts = []
    ch = first_ch[3]
    for i in range(blks):
      blk = HardBlock(ch, gr[i], grmul, n_layers[i])
      ch = blk.get_out_ch()
      skip_connection_channel_counts.append(ch)
      self.base.append(blk)
      if i < blks - 1:
        self.shortcut_layers.append(len(self.base) - 1)

      self.base.append(ConvLayer(ch_list[i], kernel=1))
      ch = ch_list[i]

      if i < blks - 1:
        self.base.append(AveragePooling2D(strides=2))

    cur_channels_count = ch
    prev_block_channels = ch
    n_blocks = blks - 1
    self.n_blocks = n_blocks

    #######################
    #   Upsampling path   #
    #######################

    self.transup_blocks = []
    self.denseup_blocks = []
    self.conv1x1_up = []

    for i in range(n_blocks - 1, -1, -1):
      self.transup_blocks.append(TransitionUp())
      cur_channels_count = prev_block_channels + skip_connection_channel_counts[
          i]
      self.conv1x1_up.append(ConvLayer(cur_channels_count // 2, kernel=1))
      cur_channels_count = cur_channels_count // 2

      blk = HardBlock(cur_channels_count, gr[i], grmul, n_layers[i])
      blk.trainable = False
      self.denseup_blocks.append(blk)
      prev_block_channels = blk.get_out_ch()

    self.final_conv = Conv2D(n_classes,
                             kernel_size=1,
                             strides=1,
                             padding='valid')

  def call(self, x):
    """Forward pass of FCHarDNet."""
    skip_connections = []
    input_shape = x.shape
    for i in range(len(self.base)):
      x = self.base[i](x)
      if i in self.shortcut_layers:
        skip_connections.append(x)
    out = x

    for i in range(self.n_blocks):
      skip = skip_connections.pop()
      out = self.transup_blocks[i](out, skip, True)
      out = self.conv1x1_up[i](out)
      out = self.denseup_blocks[i](out)

    out = self.final_conv(out)
    out = tf.image.resize(out, input_shape[1:3])
    return out[..., 0]
