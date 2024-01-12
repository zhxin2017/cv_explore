import torch
import torch.nn.functional as F


def focal_loss(logits, labels, alpha=1, gamma=2):
    seq_len, num_category = logits.shape
    softmax = F.softmax(logits, dim=-1)
    log_softmax = softmax.log()
    loss = -alpha * (1 - softmax)**gamma * log_softmax
    one_hot = torch.zeros([seq_len, num_category], device=labels.device).\
        scatter_(1, labels.to(torch.int64).unsqueeze(1), 1)
    loss = loss * one_hot
    loss = loss.sum(dim=-1)
    return loss



