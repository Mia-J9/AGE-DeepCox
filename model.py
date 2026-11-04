"""AGE-DeepCox model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RiskDecomposition:
    total_risk: Tensor
    explicit_risk: Tensor
    residual_risk: Tensor
    reference_risk: Tensor
    modification_risk: Tensor
    age_main_risk: Tensor
    gene_contributions: Tensor
    raw_nonlinear_risk: Tensor
    orthogonalized_nonlinear_risk: Tensor
    gate: Tensor


class NonlinearResidualBranch(nn.Module):
    def __init__(
        self,
        n_genes: int,
        hidden_dim: int,
        gate_init: float = 0.0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if n_genes <= 0 or hidden_dim <= 0:
            raise ValueError("n_genes and hidden_dim must be positive.")
        if not -1.0 < gate_init < 1.0:
            raise ValueError("gate_init must lie in (-1, 1).")

        self.fc1 = nn.Linear(n_genes, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.gamma = nn.Parameter(torch.atanh(torch.tensor(float(gate_init))))
        self.eps = float(eps)

        self.register_buffer("training_raw_mean", torch.zeros(()))
        self.register_buffer("training_explicit_mean", torch.zeros(()))
        self.register_buffer("training_projection_coefficient", torch.zeros(()))
        self.register_buffer("orthogonalization_is_fitted", torch.tensor(False))

    @property
    def gate(self) -> Tensor:
        return torch.tanh(self.gamma)

    def raw_output(self, gene_contributions: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(gene_contributions))).squeeze(-1)

    def _fit_orthogonalization(
        self,
        raw_output: Tensor,
        explicit_risk: Tensor,
    ) -> Tensor:
        raw_mean = raw_output.mean()
        explicit_mean = explicit_risk.mean()
        raw_centered = raw_output - raw_mean
        explicit_centered = explicit_risk - explicit_mean
        coefficient = (raw_centered * explicit_centered).mean() / (
            explicit_centered.square().mean() + self.eps
        )
        with torch.no_grad():
            self.training_raw_mean.copy_(raw_mean.detach())
            self.training_explicit_mean.copy_(explicit_mean.detach())
            self.training_projection_coefficient.copy_(coefficient.detach())
            self.orthogonalization_is_fitted.fill_(True)
        return raw_centered - coefficient * explicit_centered

    def _apply_orthogonalization(
        self,
        raw_output: Tensor,
        explicit_risk: Tensor,
    ) -> Tensor:
        if not bool(self.orthogonalization_is_fitted.item()):
            raise RuntimeError("Training orthogonalization moments are not fitted.")
        raw_centered = raw_output - self.training_raw_mean
        explicit_centered = explicit_risk - self.training_explicit_mean
        return raw_centered - self.training_projection_coefficient * explicit_centered

    def forward(
        self,
        gene_contributions: Tensor,
        explicit_risk: Tensor,
        update_moments: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        raw_output = self.raw_output(gene_contributions)
        if update_moments:
            if not self.training:
                raise RuntimeError("Cannot update moments in evaluation mode.")
            orthogonalized = self._fit_orthogonalization(raw_output, explicit_risk)
        else:
            orthogonalized = self._apply_orthogonalization(raw_output, explicit_risk)
        residual_risk = self.gate * orthogonalized
        return residual_risk, raw_output, orthogonalized


class AGEDeepCox(nn.Module):
    def __init__(
        self,
        n_genes: int,
        hidden_dim: int,
        gene_names: Sequence[str] | None = None,
        gate_init: float = 0.0,
        eps: float = 1e-12,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if n_genes <= 0:
            raise ValueError("n_genes must be positive.")
        if gene_names is not None and len(gene_names) != n_genes:
            raise ValueError("gene_names must have length n_genes.")

        self.n_genes = int(n_genes)
        self.gene_names = (
            tuple(map(str, gene_names))
            if gene_names is not None
            else tuple(f"gene_{j}" for j in range(n_genes))
        )
        self.eps = float(eps)

        self.theta_reference_tilde = nn.Parameter(torch.zeros(n_genes, dtype=dtype))
        self.theta_modification_tilde = nn.Parameter(
            torch.zeros(n_genes, dtype=dtype)
        )
        self.theta_age = nn.Parameter(torch.zeros((), dtype=dtype))
        self.nonlinear_branch = NonlinearResidualBranch(
            n_genes, hidden_dim, gate_init, eps
        ).to(dtype=dtype)

        self.register_buffer("interaction_alpha", torch.zeros(n_genes, dtype=dtype))
        self.register_buffer("interaction_scale", torch.ones(n_genes, dtype=dtype))
        self.register_buffer("interaction_transform_is_fitted", torch.tensor(False))

    def set_interaction_transform(
        self,
        alpha: Tensor,
        modification_scale: Tensor,
    ) -> None:
        alpha = torch.as_tensor(
            alpha,
            dtype=self.interaction_alpha.dtype,
            device=self.interaction_alpha.device,
        )
        scale = torch.as_tensor(
            modification_scale,
            dtype=self.interaction_scale.dtype,
            device=self.interaction_scale.device,
        )
        if alpha.shape != (self.n_genes,) or scale.shape != (self.n_genes,):
            raise ValueError(f"Interaction vectors must have shape ({self.n_genes},).")
        if not torch.isfinite(alpha).all() or not torch.isfinite(scale).all():
            raise ValueError("Interaction vectors must be finite.")
        if torch.any(scale <= self.eps):
            raise ValueError("Interaction scales must be positive.")
        with torch.no_grad():
            self.interaction_alpha.copy_(alpha)
            self.interaction_scale.copy_(scale)
            self.interaction_transform_is_fitted.fill_(True)

    def _check_transform(self) -> None:
        if not bool(self.interaction_transform_is_fitted.item()):
            raise RuntimeError("Interaction transform is not fitted.")

    @property
    def theta_modification(self) -> Tensor:
        self._check_transform()
        return self.theta_modification_tilde / self.interaction_scale

    @property
    def theta_reference(self) -> Tensor:
        self._check_transform()
        return self.theta_reference_tilde - (
            self.interaction_alpha * self.theta_modification
        )

    @property
    def gate(self) -> Tensor:
        return self.nonlinear_branch.gate

    def patient_specific_effects(self, z: Tensor) -> Tensor:
        self._validate_age(z)
        return self.theta_reference[None, :] + (
            z[:, None] * self.theta_modification[None, :]
        )

    def gene_contributions(self, x: Tensor, z: Tensor) -> Tensor:
        self._validate_inputs(x, z)
        return x * self.patient_specific_effects(z)

    def explicit_components(
        self,
        x: Tensor,
        z: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        self._validate_inputs(x, z)
        reference = x * self.theta_reference[None, :]
        modification = x * z[:, None] * self.theta_modification[None, :]
        contributions = reference + modification
        reference_risk = reference.sum(dim=1)
        modification_risk = modification.sum(dim=1)
        age_main_risk = z * self.theta_age
        explicit_risk = reference_risk + modification_risk + age_main_risk
        return (
            contributions,
            reference_risk,
            modification_risk,
            age_main_risk,
            explicit_risk,
        )

    def forward(
        self,
        x: Tensor,
        z: Tensor,
        update_orthogonalization: bool | None = None,
    ) -> RiskDecomposition:
        if update_orthogonalization is None:
            update_orthogonalization = self.training
        if update_orthogonalization and not self.training:
            raise RuntimeError("Held-out data cannot update training moments.")

        (
            contributions,
            reference_risk,
            modification_risk,
            age_main_risk,
            explicit_risk,
        ) = self.explicit_components(x, z)
        residual_risk, raw_output, orthogonalized = self.nonlinear_branch(
            contributions,
            explicit_risk,
            bool(update_orthogonalization),
        )
        return RiskDecomposition(
            total_risk=explicit_risk + residual_risk,
            explicit_risk=explicit_risk,
            residual_risk=residual_risk,
            reference_risk=reference_risk,
            modification_risk=modification_risk,
            age_main_risk=age_main_risk,
            gene_contributions=contributions,
            raw_nonlinear_risk=raw_output,
            orthogonalized_nonlinear_risk=orthogonalized,
            gate=self.gate,
        )

    @torch.no_grad()
    def coefficient_table(self) -> list[dict[str, float | str]]:
        reference = self.theta_reference.detach().cpu().tolist()
        modification = self.theta_modification.detach().cpu().tolist()
        return [
            {
                "gene": gene,
                "theta_reference": float(theta_r),
                "theta_modification": float(theta_m),
            }
            for gene, theta_r, theta_m in zip(
                self.gene_names, reference, modification
            )
        ]

    @staticmethod
    def _validate_age(z: Tensor) -> None:
        if z.ndim != 1 or not torch.isfinite(z).all():
            raise ValueError("z must be a finite one-dimensional tensor.")

    def _validate_inputs(self, x: Tensor, z: Tensor) -> None:
        if x.ndim != 2 or x.shape[1] != self.n_genes:
            raise ValueError(f"x must have shape (n, {self.n_genes}).")
        self._validate_age(z)
        if x.shape[0] != z.shape[0]:
            raise ValueError("x and z must have the same number of rows.")
        if not torch.isfinite(x).all():
            raise ValueError("x contains non-finite values.")


__all__ = ["AGEDeepCox", "NonlinearResidualBranch", "RiskDecomposition"]
