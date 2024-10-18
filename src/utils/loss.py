from typing import Optional, Union

import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    """
    Implements Focal Loss to sway the weight update towards
    the direction of harder samples to mitigate the effects of class imbalance.

    Source: https://doi.org/10.48550/arXiv.1708.02002

    Parameters
    ----------
    alpha: Optional[torch.Tensor]
        The class weights.

    gamma: Union[float, int]
        The focusing parameter.

    reduction: str
        The reduction to perform on the loss values per mini-batch.
    """


    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: Union[float, int] = 2,
        reduction: str = "mean"
        ):
        super().__init__()

        valid_reduction = ["none", "sum", "mean"]
        assert reduction in valid_reduction, f"reduction must be one of {valid_reduction}"

        self.alpha = alpha if alpha != None else torch.tensor(1.0)
        self.gamma = gamma
        self.reduction = reduction

        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self, 
        logits: torch.Tensor, 
        target: torch.Tensor
        ):

        ce_loss = self.ce(logits, target)
        pt = torch.exp(-ce_loss)
        alpha = self.alpha.to(target.device)
        alpha = alpha[target] if self.alpha.numel() > 1 else alpha

        focal_loss = alpha * ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            focal_loss = focal_loss.mean()

        if self.reduction == "sum":
            focal_loss = focal_loss.sum()

        return focal_loss
