import torch
import torch.nn.functional as F


class FocalLossV1(torch.nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha if alpha is not None else 1.0
        self.reduction = reduction

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = logpt.gather(1, target.view(-1, 1))
        pt = pt.gather(1, target.view(-1, 1))
        if self.alpha is not None and hasattr(self.alpha, "gather"):
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at.view(-1, 1)
        loss = -((1 - pt) ** self.gamma) * logpt
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
