"""Sequence-Based Implementation of the Focal Loss."""

from typing import Optional

from torch import nn
from torch.nn import functional as F
from torch import Tensor


class SequenceFocalLoss(nn.Module):
    """Implements the Focal Loss from [1]. Implementation inspired by [2].

    Adds some useful features to the torch implementation -- label smoothing, weighting,
    and so on and so forth.

    [1] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, “Focal Loss for Dense
    Object Detection.” arXiv, Feb. 07, 2018. Accessed: Jun. 08, 2023. [Online].
    Available: http://arxiv.org/abs/1708.02002

    [2] https://pytorch.org/vision/main/_modules/torchvision/ops/focal_loss.html
    """

    def __init__(
        self,
        gamma: float,
        alpha: float,
        label_smoothing: float,
        ignore_index: Optional[int],
        class_weights: Optional[Tensor] = None,
        reduction: str = "mean",
        prob_computation: str = "softmax",
    ) -> None:
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.reduction = reduction

        if class_weights is not None:
            self.weights = nn.Parameter(
                class_weights,
                requires_grad=False,
            )
        else:
            self.weights = None

        if prob_computation == "sigmoid":
            self.probability = nn.Sigmoid()
        else:
            self.probability = nn.Softmax(dim=-1)

    def forward(
        self,
        output: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Compute the Focal Loss of the output of the model w.r.t. the target tokens.

        Parameters
        ----------
        output : Tensor
            A flattened tensor of elements with shape (NT) x C, where N is the batch
            size, T is the sequence length and C is the number of classes. Each
            position of the tensor is the logit probability of class c_i at time step
            t_j.
        target : Tensor
            A flattened tensor of elements with shape (NT), where N is the batch size
            and T is the sequence length. Each position of the tensor contains the index
            of the ground truth class.

        Returns
        -------
        Tensor
            A single number representing the difference from target to source.
        """
        target_onehot = F.one_hot(target, num_classes=output.size(-1)).float()

        p = self.probability(output)

        if self.label_smoothing > 0.0:
            nclasses = output.shape[-1]
            p = ((1 - self.label_smoothing) * p) + (self.label_smoothing / nclasses)

        ce_loss = F.binary_cross_entropy(
            input=p,
            target=target_onehot,
            reduction="none",
        )
        p_t = p * target_onehot + (1 - p) * (1 - target_onehot)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        alpha_t = self.alpha * target_onehot + (1 - self.alpha) * (1 - target_onehot)
        loss = alpha_t * loss

        if self.weights is not None:
            loss = loss * self.weights[None, :].expand(p_t.size(0), -1)

        if self.ignore_index is not None:
            ignore_mask = target != self.ignore_index
            loss = loss[ignore_mask]
        if self.reduction == "mean":
            loss = loss.sum(dim=0).mean()
        else:
            loss = loss.sum(dim=0).sum()

        return loss
