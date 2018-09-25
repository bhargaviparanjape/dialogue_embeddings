import torch
import torch.nn as nn
import torch.autograd as ag


class MaskedSoftmaxAndLogSoftmax(ag.Function):
  def __init__(self, dtype = torch.FloatTensor):
    super(MaskedSoftmaxAndLogSoftmax, self).__init__()
    self._dtype = dtype

  def forward(self, xs, mask):
    maxes = torch.max(xs + torch.log(mask), 1, keepdim = True)[0]
    masked_exp_xs = torch.exp(xs - maxes) * mask
    normalization_factor = masked_exp_xs.sum(1, keepdim = True)
    probs = masked_exp_xs / normalization_factor
    log_probs = (xs - maxes - torch.log(normalization_factor)) * mask

    self.save_for_backward(probs, mask)
    return probs, log_probs

  def backward(self, grad_probs, grad_log_probs):
    probs, mask = self.saved_tensors

    num_actions = grad_probs.size()[1]
    w1 = (probs * grad_probs).unsqueeze(0).unsqueeze(-1)
    w2 = torch.eye(num_actions).type(self._dtype).unsqueeze(0)
    if grad_probs.is_cuda:
      w2 = w2.cuda()
    w2 = (w2 - probs.unsqueeze(-1))

    grad1 = torch.matmul(w2, w1).squeeze(0).squeeze(-1)

    w1 = grad_log_probs
    sw1 = (mask * grad_log_probs).sum(1, keepdim = True)
    grad2 = (w1 * mask - probs * sw1)
    return grad1 + grad2, None