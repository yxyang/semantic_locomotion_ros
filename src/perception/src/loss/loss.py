"""Collection of loss functions."""
import torch
import torch.nn
import torch.nn.functional as F


def cross_entropy2d(prediction, target, weight=None, size_average=True):
  _, c, h, w = prediction.size()
  _, ht, wt = target.size()

  if h != ht and w != wt:  # upsample labels
    prediction = F.interpolate(prediction,
                               size=(ht, wt),
                               mode="bilinear",
                               align_corners=True)

  prediction = prediction.transpose(1,
                                    2).transpose(2,
                                                 3).contiguous().view(-1, c)
  target = target.view(-1)

  loss = F.cross_entropy(prediction,
                         target,
                         weight=weight,
                         size_average=size_average,
                         ignore_index=250,
                         reduction='mean')

  return loss


def multi_scale_cross_entropy2d(prediction,
                                target,
                                loss_th,
                                weight=None,
                                size_average=True,
                                scale_weight=(1.0, 0.4)):
  if not isinstance(prediction, tuple):
    return cross_entropy2d(prediction=prediction,
                           target=target,
                           weight=weight,
                           size_average=size_average)

  K = prediction[0].size()[2] * prediction[0].size()[3] // 128
  loss = 0.0

  for i, inp in enumerate(prediction):
    loss = loss + scale_weight[i] * bootstrapped_cross_entropy2d(
        prediction=inp,
        target=target,
        min_K=K,
        loss_th=loss_th,
        weight=weight)

  return loss


def bootstrapped_cross_entropy2d(prediction,
                                 target,
                                 min_K,
                                 loss_th,
                                 weight=None):
  _, _, h, w = prediction.size()
  _, ht, wt = target.size()
  batch_size = prediction.size()[0]

  if h != ht and w != wt:  # upsample labels
    prediction = F.interpolate(prediction,
                               size=(ht, wt),
                               mode="bilinear",
                               align_corners=True)

  thresh = loss_th

  def _bootstrap_xentropy_single(prediction,
                                 target,
                                 K,
                                 thresh,
                                 weight=None):

    _, c, _, _ = prediction.size()  # n, c, h, w
    prediction = prediction.transpose(1,
                                      2).transpose(2,
                                                   3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(prediction,
                           target,
                           weight=weight,
                           reduce=False,
                           size_average=False,
                           ignore_index=250)
    sorted_loss, _ = torch.sort(loss, descending=True)

    if sorted_loss[K] > thresh:
      loss = sorted_loss[sorted_loss > thresh]
    else:
      loss = sorted_loss[:K]
    reduced_topk_loss = torch.mean(loss)

    return reduced_topk_loss

  loss = 0.0
  # Bootstrap from each image not entire batch
  for i in range(batch_size):
    loss += _bootstrap_xentropy_single(
        prediction=torch.unsqueeze(prediction[i], 0),
        target=torch.unsqueeze(target[i], 0),
        K=min_K,
        thresh=thresh,
        weight=weight,
    )
  return loss / float(batch_size)
