from abc import ABC, abstractmethod
from typing import Any, Tuple

import torch


class DenoiserScaling(ABC):
    @abstractmethod
    def __call__(
        self, sigma: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pass


class EDMScaling:
    def __init__(self, sigma_data: float = 0.5):
        self.sigma_data = sigma_data
        self.sigma_data_sq = sigma_data * sigma_data

    def __call__(
        self, sigma: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # OPTIMIZATION: Compute shared expressions once
        sigma_sq = sigma * sigma
        sigma_sq_plus_data_sq = sigma_sq + self.sigma_data_sq
        inv_sigma_sq_plus_data_sq = 1.0 / sigma_sq_plus_data_sq
        sqrt_sigma_sq_plus_data_sq = sigma_sq_plus_data_sq ** 0.5

        c_skip = self.sigma_data_sq * inv_sigma_sq_plus_data_sq
        c_out = sigma * self.sigma_data / sqrt_sigma_sq_plus_data_sq
        c_in = inv_sigma_sq_plus_data_sq * sqrt_sigma_sq_plus_data_sq  # 1 / sqrt(...) = sqrt(...) / (...)
        c_noise = 0.25 * sigma.log()
        return c_skip, c_out, c_in, c_noise


class EpsScaling:
    def __call__(
        self, sigma: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c_skip = torch.ones_like(sigma, device=sigma.device)
        c_out = -sigma
        # OPTIMIZATION: Compute shared expression once
        sqrt_sigma_sq_plus_1 = (sigma * sigma + 1.0) ** 0.5
        c_in = 1.0 / sqrt_sigma_sq_plus_1
        c_noise = sigma.clone()
        return c_skip, c_out, c_in, c_noise


class VScaling:
    def __call__(
        self, sigma: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # OPTIMIZATION: Compute shared expressions once
        sigma_sq_plus_1 = sigma * sigma + 1.0
        sqrt_sigma_sq_plus_1 = sigma_sq_plus_1 ** 0.5
        inv_sigma_sq_plus_1 = 1.0 / sigma_sq_plus_1

        c_skip = inv_sigma_sq_plus_1
        c_out = -sigma / sqrt_sigma_sq_plus_1
        c_in = 1.0 / sqrt_sigma_sq_plus_1
        c_noise = sigma.clone()
        return c_skip, c_out, c_in, c_noise


class VScalingWithEDMcNoise(DenoiserScaling):
    def __call__(
        self, sigma: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # OPTIMIZATION: Compute shared expressions once
        sigma_sq_plus_1 = sigma * sigma + 1.0
        sqrt_sigma_sq_plus_1 = sigma_sq_plus_1 ** 0.5
        inv_sigma_sq_plus_1 = 1.0 / sigma_sq_plus_1

        c_skip = inv_sigma_sq_plus_1
        c_out = -sigma / sqrt_sigma_sq_plus_1
        c_in = 1.0 / sqrt_sigma_sq_plus_1
        c_noise = 0.25 * sigma.log()
        return c_skip, c_out, c_in, c_noise


class VideoScaling:  # similar to VScaling
    def __call__(
        self, alphas_cumprod_sqrt: torch.Tensor, **additional_model_inputs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c_skip = alphas_cumprod_sqrt
        c_out = -((1 - alphas_cumprod_sqrt**2) ** 0.5)
        c_in = torch.ones_like(alphas_cumprod_sqrt, device=alphas_cumprod_sqrt.device)
        c_noise = additional_model_inputs["idx"].clone()
        return c_skip, c_out, c_in, c_noise
