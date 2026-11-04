"""Optimization for AGE-DeepCox."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from model import AGEDeepCox


@dataclass(frozen=True)
class OptimizerConfig:
    lambda_reference: float
    lambda_modification: float
    explicit_step_size: float = 1e-2
    mlp_learning_rate: float = 3e-4
    gate_learning_rate: float = 5e-4
    mlp_weight_decay: float = 1e-3
    max_iterations: int = 120
    min_iterations: int = 100
    convergence_tolerance: float = 1e-4
    max_backtracking_steps: int = 20
    backtracking_factor: float = 0.5
    gradient_clip_norm: float = 5.0

    def __post_init__(self) -> None:
        nonnegative = {
            "lambda_reference": self.lambda_reference,
            "lambda_modification": self.lambda_modification,
            "mlp_weight_decay": self.mlp_weight_decay,
        }
        positive = {
            "explicit_step_size": self.explicit_step_size,
            "mlp_learning_rate": self.mlp_learning_rate,
            "gate_learning_rate": self.gate_learning_rate,
            "convergence_tolerance": self.convergence_tolerance,
            "gradient_clip_norm": self.gradient_clip_norm,
        }
        if any(value < 0 for value in nonnegative.values()):
            raise ValueError("Regularization parameters must be nonnegative.")
        if any(value <= 0 for value in positive.values()):
            raise ValueError("Learning rates and tolerances must be positive.")
        if self.max_iterations <= 0 or self.min_iterations < 0:
            raise ValueError("Iteration counts are invalid.")
        if self.min_iterations > self.max_iterations:
            raise ValueError("min_iterations cannot exceed max_iterations.")
        if self.max_backtracking_steps < 0:
            raise ValueError("max_backtracking_steps must be nonnegative.")
        if not 0 < self.backtracking_factor < 1:
            raise ValueError("backtracking_factor must lie in (0, 1).")


@dataclass(frozen=True)
class FitSummary:
    iterations: int
    converged: bool
    objective: float
    cox_loss: float
    reference_penalty: float
    modification_penalty: float
    accepted_explicit_step_size: float
    gate: float


def cox_breslow_loss(risk: Tensor, time: Tensor, event: Tensor) -> Tensor:
    if risk.ndim != 1 or time.ndim != 1 or event.ndim != 1:
        raise ValueError("risk, time, and event must be one-dimensional.")
    if not (risk.shape == time.shape == event.shape):
        raise ValueError("risk, time, and event must have identical shapes.")
    if not torch.isfinite(risk).all() or not torch.isfinite(time).all():
        raise ValueError("risk and time must be finite.")
    if torch.any(time <= 0):
        raise ValueError("Survival times must be positive.")

    event_mask = event > 0.5
    n_events = event_mask.sum()
    if n_events == 0:
        raise ValueError("At least one observed event is required.")

    loss = risk.new_zeros(())
    for event_time in torch.unique(time[event_mask]):
        tied = event_mask & (time == event_time)
        at_risk = time >= event_time
        n_tied = tied.sum().to(risk.dtype)
        loss -= risk[tied].sum() - n_tied * torch.logsumexp(risk[at_risk], dim=0)
    return loss / n_events.to(risk.dtype)


@torch.no_grad()
def hierarchical_soft_threshold(
    theta_reference: Tensor,
    theta_modification: Tensor,
    reference_threshold: float,
    modification_threshold: float,
) -> tuple[Tensor, Tensor]:
    reference_magnitude = theta_reference.abs() - reference_threshold
    modification_magnitude = theta_modification.abs() - modification_threshold
    pooled = torch.clamp(
        (reference_magnitude + modification_magnitude) / 2.0,
        min=0.0,
    )
    ordered = reference_magnitude >= modification_magnitude
    reference_projected = torch.where(
        ordered,
        torch.clamp(reference_magnitude, min=0.0),
        pooled,
    )
    modification_projected = torch.where(
        ordered,
        torch.minimum(
            torch.clamp(modification_magnitude, min=0.0),
            reference_projected,
        ),
        pooled,
    )
    return (
        theta_reference.sign() * reference_projected,
        theta_modification.sign() * modification_projected,
    )


class AGEDeepCoxOptimizer:
    def __init__(self, model: AGEDeepCox, config: OptimizerConfig) -> None:
        self.model = model
        self.config = config
        self.neural_optimizer = torch.optim.Adam(
            [
                {
                    "params": list(model.nonlinear_branch.fc1.parameters())
                    + list(model.nonlinear_branch.fc2.parameters()),
                    "lr": config.mlp_learning_rate,
                    "weight_decay": config.mlp_weight_decay,
                },
                {
                    "params": [model.nonlinear_branch.gamma],
                    "lr": config.gate_learning_rate,
                },
            ]
        )

    def _penalties(self) -> tuple[Tensor, Tensor]:
        reference = self.config.lambda_reference * self.model.theta_reference.abs().sum()
        modification = (
            self.config.lambda_modification
            * self.model.theta_modification.abs().sum()
        )
        return reference, modification

    def _objective(
        self,
        x: Tensor,
        z: Tensor,
        time: Tensor,
        event: Tensor,
        update_orthogonalization: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        output = self.model(
            x,
            z,
            update_orthogonalization=update_orthogonalization,
        )
        cox = cox_breslow_loss(output.total_risk, time, event)
        reference_penalty, modification_penalty = self._penalties()
        total = cox + reference_penalty + modification_penalty
        return total, cox, reference_penalty, modification_penalty

    @torch.no_grad()
    def _set_interpretable_coefficients(
        self,
        theta_reference: Tensor,
        theta_modification: Tensor,
        theta_age: Tensor,
    ) -> None:
        theta_modification_tilde = (
            theta_modification * self.model.interaction_scale
        )
        theta_reference_tilde = theta_reference + (
            self.model.interaction_alpha * theta_modification
        )
        self.model.theta_reference_tilde.copy_(theta_reference_tilde)
        self.model.theta_modification_tilde.copy_(theta_modification_tilde)
        self.model.theta_age.copy_(theta_age)

    def _explicit_step(
        self,
        x: Tensor,
        z: Tensor,
        time: Tensor,
        event: Tensor,
    ) -> float:
        self.model.train()
        current, current_cox, _, _ = self._objective(x, z, time, event, True)
        parameters = (
            self.model.theta_reference_tilde,
            self.model.theta_modification_tilde,
            self.model.theta_age,
        )
        gradients = torch.autograd.grad(current_cox, parameters)
        if not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise FloatingPointError("Non-finite explicit-parameter gradient.")

        old_reference_tilde = parameters[0].detach().clone()
        old_modification_tilde = parameters[1].detach().clone()
        old_age = parameters[2].detach().clone()
        accepted_step = 0.0

        for attempt in range(self.config.max_backtracking_steps + 1):
            step = self.config.explicit_step_size * (
                self.config.backtracking_factor**attempt
            )
            candidate_reference_tilde = old_reference_tilde - step * gradients[0]
            candidate_modification_tilde = (
                old_modification_tilde - step * gradients[1]
            )
            candidate_age = old_age - step * gradients[2]

            candidate_modification = (
                candidate_modification_tilde / self.model.interaction_scale
            )
            candidate_reference = candidate_reference_tilde - (
                self.model.interaction_alpha * candidate_modification
            )
            candidate_reference, candidate_modification = hierarchical_soft_threshold(
                candidate_reference,
                candidate_modification,
                step * self.config.lambda_reference,
                step * self.config.lambda_modification,
            )
            self._set_interpretable_coefficients(
                candidate_reference,
                candidate_modification,
                candidate_age,
            )
            with torch.no_grad():
                candidate, _, _, _ = self._objective(x, z, time, event, True)
            if torch.isfinite(candidate) and candidate <= current.detach() + 1e-10:
                accepted_step = step
                break
        else:
            with torch.no_grad():
                self.model.theta_reference_tilde.copy_(old_reference_tilde)
                self.model.theta_modification_tilde.copy_(old_modification_tilde)
                self.model.theta_age.copy_(old_age)

        return accepted_step

    def _neural_step(
        self,
        x: Tensor,
        z: Tensor,
        time: Tensor,
        event: Tensor,
    ) -> None:
        self.neural_optimizer.zero_grad(set_to_none=True)
        output = self.model(x, z, update_orthogonalization=True)
        loss = cox_breslow_loss(output.total_risk, time, event)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite nonlinear-branch loss.")
        loss.backward()
        neural_parameters = [
            parameter
            for group in self.neural_optimizer.param_groups
            for parameter in group["params"]
        ]
        torch.nn.utils.clip_grad_norm_(
            neural_parameters,
            self.config.gradient_clip_norm,
        )
        if not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in neural_parameters
        ):
            raise FloatingPointError("Non-finite nonlinear-branch gradient.")
        self.neural_optimizer.step()
        self.model.theta_reference_tilde.grad = None
        self.model.theta_modification_tilde.grad = None
        self.model.theta_age.grad = None

    def fit(
        self,
        x: Tensor,
        z: Tensor,
        time: Tensor,
        event: Tensor,
    ) -> FitSummary:
        self._validate_training_data(x, z, time, event)
        previous_objective: float | None = None
        converged = False
        accepted_step = 0.0

        for iteration in range(1, self.config.max_iterations + 1):
            accepted_step = self._explicit_step(x, z, time, event)
            self._neural_step(x, z, time, event)

            self.model.train()
            with torch.no_grad():
                total, cox, reference_penalty, modification_penalty = self._objective(
                    x, z, time, event, True
                )
            values = (total, cox, reference_penalty, modification_penalty)
            if not all(torch.isfinite(value) for value in values):
                raise FloatingPointError("Non-finite training objective.")

            objective = float(total)
            if previous_objective is not None and iteration >= self.config.min_iterations:
                relative_change = abs(objective - previous_objective) / max(
                    abs(previous_objective), 1e-12
                )
                if relative_change < self.config.convergence_tolerance:
                    converged = True
                    break
            previous_objective = objective

        self.model.train()
        with torch.no_grad():
            total, cox, reference_penalty, modification_penalty = self._objective(
                x, z, time, event, True
            )
        self.model.eval()
        return FitSummary(
            iterations=iteration,
            converged=converged,
            objective=float(total),
            cox_loss=float(cox),
            reference_penalty=float(reference_penalty),
            modification_penalty=float(modification_penalty),
            accepted_explicit_step_size=float(accepted_step),
            gate=float(self.model.gate.detach()),
        )

    @staticmethod
    def _validate_training_data(
        x: Tensor,
        z: Tensor,
        time: Tensor,
        event: Tensor,
    ) -> None:
        n = x.shape[0]
        if z.shape != (n,) or time.shape != (n,) or event.shape != (n,):
            raise ValueError("Training tensors have inconsistent shapes.")
        tensors = (x, z, time, event)
        if not all(torch.isfinite(tensor).all() for tensor in tensors):
            raise ValueError("Training tensors contain non-finite values.")
        if not torch.all((event == 0) | (event == 1)):
            raise ValueError("event must be coded as 0/1.")


__all__ = [
    "AGEDeepCoxOptimizer",
    "FitSummary",
    "OptimizerConfig",
    "cox_breslow_loss",
    "hierarchical_soft_threshold",
]
