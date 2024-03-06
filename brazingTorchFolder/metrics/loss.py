import torch


class quantileLoss(torch.nn.Module):
    def __init__(self, quantiles):
        self.quantiles = quantiles

    def __call__(self, yPred, target):
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - yPred[..., i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
        losses = 2 * torch.cat(losses, dim=2).mean()

        return losses
