"""DDPM-style schedules and helper functions for latent diffusion."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class DiffusionScheduleConfig:
    timesteps: int = 1000
    beta_start: float = 0.00085
    beta_end: float = 0.012
    beta_schedule: str = "linear"
    cosine_s: float = 0.008
    prediction_type: str = "epsilon"
    loss_weighting: str = "uniform"
    min_snr_gamma: float = 5.0


class DiffusionSchedule:
    def __init__(self, config: DiffusionScheduleConfig, device: torch.device | str = "cpu") -> None:
        self.config = config
        self.device = torch.device(device)
        self.betas = self._make_betas(config).to(self.device)
        self.alphas = (1.0 - self.betas).to(self.device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), self.alphas_cumprod[:-1]], dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt((1.0 / self.alphas_cumprod) - 1.0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        ).clamp_min(1e-20)
        self.snr = self.alphas_cumprod / (1.0 - self.alphas_cumprod).clamp_min(1e-20)

    @staticmethod
    def _make_betas(config: DiffusionScheduleConfig) -> torch.Tensor:
        if config.beta_schedule == "linear":
            return torch.linspace(config.beta_start, config.beta_end, config.timesteps, dtype=torch.float32)
        if config.beta_schedule == "cosine":
            betas = []
            timesteps = config.timesteps
            offset = config.cosine_s
            for step in range(timesteps):
                t1 = step / timesteps
                t2 = (step + 1) / timesteps
                alpha_bar_t1 = math.cos(((t1 + offset) / (1.0 + offset)) * math.pi * 0.5) ** 2
                alpha_bar_t2 = math.cos(((t2 + offset) / (1.0 + offset)) * math.pi * 0.5) ** 2
                betas.append(min(1.0 - (alpha_bar_t2 / max(alpha_bar_t1, 1e-12)), 0.999))
            return torch.tensor(betas, dtype=torch.float32)
        raise RuntimeError(f"Unsupported beta schedule: {config.beta_schedule}")

    def sample_timesteps(self, batch_size: int, *, device: torch.device | None = None) -> torch.Tensor:
        return torch.randint(0, self.config.timesteps, (batch_size,), device=device or self.device, dtype=torch.long)

    def _extract(self, values: torch.Tensor, timesteps: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        gathered = values.gather(0, timesteps)
        return gathered.view(target_shape[0], *([1] * (len(target_shape) - 1)))

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            self._extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * x_start
            + self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape) * noise
        )

    def compute_velocity(self, x_start: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return (
            self._extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * noise
            - self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape) * x_start
        )

    def predict_epsilon(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        prediction_type: str = "epsilon",
    ) -> torch.Tensor:
        if prediction_type == "epsilon":
            return model_output
        if prediction_type == "v_prediction":
            return (
                self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_t.shape) * x_t
                + self._extract(self.sqrt_alphas_cumprod, timesteps, x_t.shape) * model_output
            )
        raise RuntimeError(f"Unsupported diffusion prediction_type: {prediction_type}")

    def compute_snr(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.snr.gather(0, timesteps)

    def p_mean_variance(self, x_t: torch.Tensor, timesteps: torch.Tensor, eps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        betas_t = self._extract(self.betas, timesteps, x_t.shape)
        sqrt_one_minus_alpha_bar = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_t.shape)
        sqrt_recip_alpha = self._extract(self.sqrt_recip_alphas, timesteps, x_t.shape)
        model_mean = sqrt_recip_alpha * (x_t - betas_t * eps / sqrt_one_minus_alpha_bar.clamp_min(1e-8))
        posterior_variance = self._extract(self.posterior_variance, timesteps, x_t.shape)
        return model_mean, posterior_variance

    @torch.no_grad()
    def sample_loop(
        self,
        model,
        shape: torch.Size,
        context_tokens: torch.Tensor,
        *,
        guidance_scale: float = 1.0,
        null_context: torch.Tensor | None = None,
        prediction_type: str = "epsilon",
        device: torch.device | None = None,
    ) -> torch.Tensor:
        device = device or self.device
        x_t = torch.randn(shape, device=device)
        for step in reversed(range(self.config.timesteps)):
            timesteps = torch.full((shape[0],), step, device=device, dtype=torch.long)
            if guidance_scale != 1.0 and null_context is not None:
                eps_uncond = self.predict_epsilon(
                    model(x_t, timesteps, null_context),
                    x_t,
                    timesteps,
                    prediction_type=prediction_type,
                )
                eps_cond = self.predict_epsilon(
                    model(x_t, timesteps, context_tokens),
                    x_t,
                    timesteps,
                    prediction_type=prediction_type,
                )
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                eps = self.predict_epsilon(
                    model(x_t, timesteps, context_tokens),
                    x_t,
                    timesteps,
                    prediction_type=prediction_type,
                )
            model_mean, posterior_variance = self.p_mean_variance(x_t, timesteps, eps)
            noise = torch.randn_like(x_t)
            nonzero_mask = (timesteps > 0).float().view(shape[0], *([1] * (x_t.ndim - 1)))
            x_t = model_mean + nonzero_mask * torch.sqrt(posterior_variance) * noise
        return x_t
