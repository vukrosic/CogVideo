from abc import abstractmethod
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        # OPTIMIZATION: Compute var first, then std = sqrt(var) avoids extra exp()
        self.var = torch.exp(self.logvar)
        self.std = self.var.sqrt()
        if self.deterministic:
            # OPTIMIZATION: Removed redundant .to(device=...) - zeros_like inherits device
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self):
        # x = self.mean + self.std * torch.randn(self.mean.shape).to(
        #     device=self.parameters.device
        # )
        x = self.mean + self.std * torch.randn_like(self.mean)
        return x

    def kl(self, other=None):
        if self.deterministic:
            # OPTIMIZATION: Use tensor on correct device instead of CPU tensor
            return torch.zeros(1, dtype=self.mean.dtype, device=self.mean.device)
        else:
            if other is None:
                # OPTIMIZATION: Use ** 2 instead of torch.pow(x, 2)
                return 0.5 * torch.sum(
                    self.mean ** 2 + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                return 0.5 * torch.sum(
                    (self.mean - other.mean) ** 2 / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            # OPTIMIZATION: Use tensor on correct device instead of CPU tensor
            return torch.zeros(1, dtype=self.mean.dtype, device=self.mean.device)
        # OPTIMIZATION: Precompute constant, use ** 2 instead of torch.pow
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + (sample - self.mean) ** 2 / self.var,
            dim=dims,
        )

    def mode(self):
        return self.mean


class AbstractRegularizer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        raise NotImplementedError()

    @abstractmethod
    def get_trainable_parameters(self) -> Any:
        raise NotImplementedError()


class IdentityRegularizer(AbstractRegularizer):
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        return z, dict()

    def get_trainable_parameters(self) -> Any:
        yield from ()


def measure_perplexity(
    predicted_indices: torch.Tensor, num_centroids: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    # src: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py
    # eval cluster perplexity. when perplexity == num_embeddings then all clusters are used exactly equally
    encodings = F.one_hot(predicted_indices, num_centroids).float().reshape(-1, num_centroids)
    avg_probs = encodings.mean(0)
    perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
    cluster_use = torch.sum(avg_probs > 0)
    return perplexity, cluster_use


class DiagonalGaussianRegularizer(AbstractRegularizer):
    def __init__(self, sample: bool = True):
        super().__init__()
        self.sample = sample

    def get_trainable_parameters(self) -> Any:
        yield from ()

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        log = dict()
        posterior = DiagonalGaussianDistribution(z)
        if self.sample:
            z = posterior.sample()
        else:
            z = posterior.mode()
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        log["kl_loss"] = kl_loss
        return z, log
