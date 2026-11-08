from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from pinn_starlight_core.data.image_loader import ImageLoader, rgb_to_gray
from pinn_starlight_core.physics.forcing import candidate_centers_from_observed


class Alpha(nn.Module):
    def __init__(self, init: float = 1.0, alpha_min: float = 0.5, alpha_max: float = 1.5):
        super().__init__()
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.init = init

        ratio = (init - alpha_min) / (alpha_max - alpha_min)
        ratio = torch.tensor(ratio, dtype=torch.float32).clamp(1e-6, 1 - 1e-6)
        self.raw_alpha = nn.Parameter(torch.logit(ratio))

    def forward(self):
        length = self.alpha_max - self.alpha_min
        ratio = torch.sigmoid(self.raw_alpha)
        return self.alpha_min + length * ratio

    def get_str(self):
        return f"[min:{self.alpha_min},max:{self.alpha_max},init:{self.init}]"


def _point_source_term(r: torch.Tensor, alpha: torch.Tensor):
    """Signed q(r; alpha) forcing template induced by the radial ansatz."""
    safe_r = torch.sqrt(r * r + 1e-8)
    sqrt_alpha = torch.sqrt(alpha)
    cos_term = 2 * alpha * torch.cos(safe_r * sqrt_alpha)
    sin_term = (sqrt_alpha / safe_r) * torch.sin(safe_r * sqrt_alpha)
    quad_term = -16 * (safe_r**2) * (alpha**2)
    quartic_term = (safe_r**4) * (alpha**3)
    return cos_term + sin_term + quad_term + quartic_term


def _inverse_softplus(value: float) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError("softplus initialization must be positive")
    return math.log(math.expm1(value))


def _load_gray_image(loader: ImageLoader):
    return rgb_to_gray(loader.rgb_data)


class EffectiveForcingMixture(nn.Module):
    """Sparse K-component signed forcing model shared conceptually with cosine-domain LS.

    The components are *effective forcing templates*, not direct non-negative
    urban radiance measurements. Each template may have a negative lobe after
    the screened-Poisson differential operator is applied to the radial ansatz.
    Non-negative weights therefore mean non-negative template amplitudes, not
    pointwise non-negative forcing.
    """

    def __init__(
        self,
        device,
        *,
        loader: ImageLoader | None = None,
        gray_img: np.ndarray | None = None,
        max_components: int = 3,
        sigma_min: float = 0.05,
        sigma_max: float = 0.55,
        theta_bound: float = 0.5 * math.pi,
        winsor_percentile: float = 99.5,
        low_pass_sigma_fraction: float = 0.04,
        min_center_distance: float = 0.30,
        center_refinement: float = 0.08,
        center_refinement_enabled: bool = True,
        initial_weights=(1.0, 0.05, 0.05),
        initial_sigma: float = 0.425,
        initial_centers: np.ndarray | None = None,
        rms_grid_size: int = 64,
        rms_epsilon: float = 1e-8,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.max_components = int(max_components)
        if self.max_components <= 0:
            raise ValueError("max_components must be positive")
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        if not 0.0 < self.sigma_min < self.sigma_max:
            raise ValueError("sigma bounds must satisfy 0 < min < max")
        self.sigma_scale = self.sigma_max - self.sigma_min
        self.theta_bound = float(theta_bound)
        self.center_refinement = float(center_refinement)
        self.center_refinement_enabled = bool(center_refinement_enabled)
        self.rms_grid_size = int(rms_grid_size)
        self.rms_epsilon = float(rms_epsilon)

        if gray_img is None:
            if loader is None:
                raise ValueError("loader and gray_img cannot both be None")
            gray_img = _load_gray_image(loader)
        gray_array = np.asarray(gray_img, dtype=np.float32)
        if gray_array.ndim != 2 or not np.isfinite(gray_array).all():
            raise ValueError("gray_img must be a finite 2-D array")
        self.H, self.W = gray_array.shape

        if initial_centers is None:
            initial_centers = candidate_centers_from_observed(
                gray_array,
                max_components=self.max_components,
                winsor_percentile=float(winsor_percentile),
                low_pass_sigma_fraction=float(low_pass_sigma_fraction),
                min_center_distance=float(min_center_distance),
            )
        else:
            initial_centers = np.asarray(initial_centers, dtype=np.float64)
            if initial_centers.shape != (self.max_components, 2):
                raise ValueError(
                    "initial_centers must have shape "
                    f"({self.max_components}, 2), got {initial_centers.shape}"
                )
            if not np.isfinite(initial_centers).all():
                raise ValueError("initial_centers must be finite")
            if np.any(initial_centers < -1.0) or np.any(initial_centers > 1.0):
                raise ValueError("initial_centers must lie in normalized [-1, 1] coordinates")
        self.register_buffer(
            "initial_centers",
            torch.tensor(initial_centers, dtype=torch.float32, device=self.device),
        )
        self.raw_center_offsets = nn.Parameter(
            torch.zeros((self.max_components, 2), dtype=torch.float32, device=self.device),
            requires_grad=self.center_refinement_enabled,
        )
        initial_sigma = float(initial_sigma)
        if not self.sigma_min < initial_sigma < self.sigma_max:
            raise ValueError("initial_sigma must lie strictly inside sigma bounds")
        sigma_fraction = (initial_sigma - self.sigma_min) / self.sigma_scale
        raw_sigma_init = math.log(sigma_fraction / (1.0 - sigma_fraction))
        self.raw_sigma_x = nn.Parameter(
            torch.full(
                (self.max_components,), raw_sigma_init, dtype=torch.float32, device=self.device
            )
        )
        self.raw_sigma_y = nn.Parameter(
            torch.full(
                (self.max_components,), raw_sigma_init, dtype=torch.float32, device=self.device
            )
        )
        self.raw_theta = nn.Parameter(
            torch.zeros(self.max_components, dtype=torch.float32, device=self.device)
        )

        initial_weights = tuple(float(value) for value in initial_weights)
        if len(initial_weights) < self.max_components:
            raise ValueError("initial_weights must cover max_components")
        raw_weights = [_inverse_softplus(value) for value in initial_weights[: self.max_components]]
        self.raw_weights = nn.Parameter(
            torch.tensor(raw_weights, dtype=torch.float32, device=self.device)
        )

        axis = torch.linspace(-1.0, 1.0, self.rms_grid_size, device=self.device)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        canonical = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
        self.register_buffer("canonical_grid", canonical)
        self._center_grad_enabled = True

    def set_center_gradient_enabled(self, enabled: bool) -> None:
        """Runtime warm-up gate; does not alter the configured center bounds."""
        self._center_grad_enabled = bool(enabled)

    def get_centers(self):
        if not self.center_refinement_enabled:
            return self.initial_centers
        offsets = self.center_refinement * torch.tanh(self.raw_center_offsets)
        if not self._center_grad_enabled:
            offsets = offsets.detach()
        return torch.clamp(self.initial_centers + offsets, -1.0, 1.0)

    def get_sigmas(self):
        sigma_x = self.sigma_min + self.sigma_scale * torch.sigmoid(self.raw_sigma_x)
        sigma_y = self.sigma_min + self.sigma_scale * torch.sigmoid(self.raw_sigma_y)
        return sigma_x, sigma_y

    # Stage-9 compatibility helper: report the strongest component's two sigmas.
    def get_sigma(self):
        sigma_x, sigma_y = self.get_sigmas()
        index = int(torch.argmax(self.get_weights().detach()).item())
        return sigma_x[index : index + 1], sigma_y[index : index + 1]

    def get_thetas(self):
        return self.theta_bound * torch.tanh(self.raw_theta)

    def get_theta(self):
        theta = self.get_thetas()
        index = int(torch.argmax(self.get_weights().detach()).item())
        return theta[index : index + 1]

    def get_weights(self):
        return torch.nn.functional.softplus(self.raw_weights)

    def sparse_penalty(self):
        return self.get_weights().mean()

    def _unnormalized_components(self, coords, alpha):
        x = coords[:, 0].unsqueeze(1)
        y = coords[:, 1].unsqueeze(1)
        centers = self.get_centers()
        dx = x - centers[:, 0].unsqueeze(0)
        dy = y - centers[:, 1].unsqueeze(0)
        r = torch.sqrt(dx * dx + dy * dy + 1e-8)

        point_source = _point_source_term(r, alpha)
        sigma_x, sigma_y = self.get_sigmas()
        theta = self.get_thetas()
        cos_theta = torch.cos(theta).unsqueeze(0)
        sin_theta = torch.sin(theta).unsqueeze(0)
        dx_rot = cos_theta * dx + sin_theta * dy
        dy_rot = -sin_theta * dx + cos_theta * dy
        envelope = torch.exp(
            -0.5
            * (
                (dx_rot / sigma_x.unsqueeze(0)) ** 2
                + (dy_rot / sigma_y.unsqueeze(0)) ** 2
            )
        )
        return point_source * envelope

    def component_rms(self, alpha):
        canonical = self._unnormalized_components(self.canonical_grid, alpha)
        return torch.sqrt(
            torch.mean(canonical * canonical, dim=0) + self.rms_epsilon
        )

    def forward(self, coords, alpha):
        components = self._unnormalized_components(coords, alpha)
        normalized = components / self.component_rms(alpha).unsqueeze(0)
        return torch.sum(normalized * self.get_weights().unsqueeze(0), dim=1)

    @torch.no_grad()
    def diagnostics(self, alpha, inactive_relative_threshold: float = 0.05):
        centers = self.get_centers().detach()
        sigma_x, sigma_y = self.get_sigmas()
        theta = self.get_thetas()
        weights = self.get_weights()
        canonical = self._unnormalized_components(self.canonical_grid, alpha)
        rms = torch.sqrt(torch.mean(canonical * canonical, dim=0) + self.rms_epsilon)
        normalized = canonical / rms.unsqueeze(0)
        negative_energy = torch.sum(torch.minimum(normalized, torch.zeros_like(normalized)) ** 2, dim=0)
        total_energy = torch.sum(normalized**2, dim=0).clamp_min(1e-12)
        negative_fraction = negative_energy / total_energy
        relative = weights / weights.max().clamp_min(1e-12)
        # One host transfer avoids dozens of scalar CUDA synchronizations at
        # every history checkpoint while preserving the exact diagnostics.
        table = torch.stack(
            (
                self.initial_centers[:, 0],
                self.initial_centers[:, 1],
                centers[:, 0],
                centers[:, 1],
                sigma_x,
                sigma_y,
                theta,
                weights,
                relative,
                rms,
                negative_fraction,
            ),
            dim=1,
        ).detach().cpu().numpy()
        components = []
        for index, row in enumerate(table):
            components.append(
                {
                    "index": index + 1,
                    "initial_center_x": float(row[0]),
                    "initial_center_y": float(row[1]),
                    "center_x": float(row[2]),
                    "center_y": float(row[3]),
                    "sigma_x": float(row[4]),
                    "sigma_y": float(row[5]),
                    "theta": float(row[6]),
                    "weight": float(row[7]),
                    "relative_weight": float(row[8]),
                    "active": bool(row[8] >= inactive_relative_threshold),
                    "canonical_rms_before_normalization": float(row[9]),
                    "negative_energy_fraction": float(row[10]),
                }
            )
        return components


# Compatibility alias for older imports. Stage 10 documentation uses
# EffectiveForcingMixture / S_city terminology rather than I_city radiance.
Icity = EffectiveForcingMixture
