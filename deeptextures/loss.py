"""Loss functions.
"""
import torch
from torch.nn.functional import mse_loss


def gramm(tnsr: torch.Tensor) -> torch.Tensor:
    """Computes Gram matrix for the input batch tensor.

    Args:
        tnsr (torch.Tensor): input tensor of the Size([B, C, H, W]).

    Returns:
        G (torch.Tensor): output tensor of the Size([B, C, C]).
    """
    N = tnsr.size(1)  # Number of filters (size of feature maps) = C
    M = tnsr.size(-2) * tnsr.size(-1)  # size of the feature maps = H * W
    # Feature map: B, C, H, W ->  B, N, M
    F = tnsr.view(tnsr.size(0), N, M)  # Size([B, N, M])
    # Gram matrix: B, N, M -> B, N, N
    G = F.bmm(F.transpose(1, 2))  # Size([B, N, N])
    return G


def gram_loss(input: torch.Tensor, target: torch.Tensor, weight: float = 1.0):
    """Computes MSE Loss for 2 Gram matrices of the same type.

    Args:
        input (torch.Tensor):
        target (torch.Tensor):
        weight (float):

    Returns
        loss (torch.Tensor): computed loss value.
    """
    Bi, Bt = input.size(0), target.size(0)
    assert Bi == Bt

    Ni, Nt = input.size(1), target.size(1)
    assert Ni == Nt

    Mi, Mt = input.size(-2) * input.size(-1), target.size(-2) * target.size(-1)
    assert Mi == Mt

    B, N, M = Bi, Ni, Mi

    Gi, Gt = gramm(input), gramm(target)

    loss = weight * (1 / (4 * N ** 2 * M ** 2)) * mse_loss(Gi, Gt, reduction="sum") / B
    return loss
