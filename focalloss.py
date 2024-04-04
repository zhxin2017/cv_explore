import torch
import torch.nn.functional as F


def focal_loss(logits, labels, alpha=1, gamma=4):
    seq_len, num_category = logits.shape
    softmax = F.softmax(logits, dim=-1)
    log_softmax = softmax.log()
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.unsqueeze(0)
    loss = -alpha * (1 - softmax) ** gamma * log_softmax
    one_hot = torch.zeros([seq_len, num_category], device=labels.device). \
        scatter_(1, labels.to(torch.int64).unsqueeze(1), 1)
    loss = loss * one_hot
    loss = loss.sum(dim=-1)
    return loss


def cal_weights(occurrence, recover=1):
    no_zero_mask = occurrence > 0
    occurrence = occurrence + 1e-5 * (1 - no_zero_mask * 1)
    harmonic_mean = no_zero_mask.sum() / ((1 / occurrence) * no_zero_mask).sum()
    weights = (harmonic_mean / occurrence) * no_zero_mask
    return weights**recover
