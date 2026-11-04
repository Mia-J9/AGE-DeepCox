"""Training entry points for AGE-DeepCox."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch import Tensor

try:
    import yaml
except ImportError:
    yaml = None

from model import AGEDeepCox
from optimizer import AGEDeepCoxOptimizer, OptimizerConfig


@dataclass(frozen=True)
class DataConfig:
    sample_column: str = "sampleID"
    time_column: str = "time"
    event_column: str = "status"
    age_column: str = "age"
    validation_fraction: float = 0.30
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must lie in (0, 1).")
        if self.eps <= 0:
            raise ValueError("eps must be positive.")


@dataclass(frozen=True)
class PreprocessingState:
    expression_mean: np.ndarray
    expression_scale: np.ndarray
    age_mean: float
    age_scale: float
    interaction_alpha: np.ndarray
    interaction_scale: np.ndarray


@dataclass(frozen=True)
class PreparedSplit:
    train_ids: np.ndarray
    validation_ids: np.ndarray
    x_train: Tensor
    z_train: Tensor
    time_train: Tensor
    event_train: Tensor
    x_validation: Tensor
    z_validation: Tensor
    time_validation: Tensor
    event_validation: Tensor
    state: PreprocessingState
    gene_names: tuple[str, ...]


@dataclass(frozen=True)
class RunResult:
    run: int
    seed: int
    iterations: int
    converged: bool
    objective: float
    gate: float
    checkpoint: str
    predictions: str
    coefficients: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_tables(
    expression_path: str | Path,
    clinical_path: str | Path,
    config: DataConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    expression = pd.read_csv(expression_path)
    clinical = pd.read_csv(clinical_path)
    required = {
        config.sample_column,
        config.time_column,
        config.event_column,
        config.age_column,
    }
    missing = required.difference(clinical.columns)
    if missing:
        raise ValueError(f"Clinical table is missing columns: {sorted(missing)}")
    if config.sample_column not in expression.columns:
        raise ValueError(f"Expression table is missing {config.sample_column!r}.")
    if expression[config.sample_column].duplicated().any():
        raise ValueError("Expression sample identifiers must be unique.")
    if clinical[config.sample_column].duplicated().any():
        raise ValueError("Clinical sample identifiers must be unique.")

    gene_names = tuple(
        str(column)
        for column in expression.columns
        if column != config.sample_column
    )
    if not gene_names:
        raise ValueError("Expression table contains no gene columns.")

    merged = clinical[list(required)].merge(
        expression,
        on=config.sample_column,
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("The expression and clinical tables share no samples.")
    numeric_columns = [
        config.time_column,
        config.event_column,
        config.age_column,
        *gene_names,
    ]
    merged[numeric_columns] = merged[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if merged[numeric_columns].isna().any().any():
        raise ValueError("Model input contains missing or non-numeric values.")
    if (merged[config.time_column] <= 0).any():
        raise ValueError("Survival times must be positive.")
    if not merged[config.event_column].isin([0, 1]).all():
        raise ValueError("event must be coded as 0/1.")
    return merged, expression, gene_names


def make_split(
    frame: pd.DataFrame,
    config: DataConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    event = frame[config.event_column].to_numpy(dtype=int)
    time = frame[config.time_column].to_numpy(dtype=float)
    median = float(np.median(time))
    strata = event.astype(str) + "_" + (time >= median).astype(int).astype(str)
    counts = pd.Series(strata).value_counts()
    if counts.min() < 2:
        strata = event
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=config.validation_fraction,
        random_state=seed,
    )
    try:
        train_index, validation_index = next(splitter.split(frame, strata))
    except ValueError:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(frame))
        n_validation = max(1, round(len(frame) * config.validation_fraction))
        validation_index = order[:n_validation]
        train_index = order[n_validation:]
    return np.sort(train_index), np.sort(validation_index)


def fit_preprocessing(
    expression: np.ndarray,
    age: np.ndarray,
    eps: float,
) -> PreprocessingState:
    expression_mean = expression.mean(axis=0)
    expression_scale = expression.std(axis=0, ddof=0)
    expression_scale = np.where(expression_scale > eps, expression_scale, 1.0)
    age_mean = float(age.mean())
    age_scale = float(age.std(ddof=0))
    if age_scale <= eps:
        raise ValueError("Training age has zero variance.")

    x = (expression - expression_mean) / expression_scale
    z = (age - age_mean) / age_scale
    interaction = x * z[:, None]
    alpha = (x * interaction).sum(axis=0) / (np.square(x).sum(axis=0) + eps)
    residualized = interaction - x * alpha
    interaction_scale = residualized.std(axis=0, ddof=0)
    interaction_scale = np.where(interaction_scale > eps, interaction_scale, 1.0)
    return PreprocessingState(
        expression_mean=expression_mean,
        expression_scale=expression_scale,
        age_mean=age_mean,
        age_scale=age_scale,
        interaction_alpha=alpha,
        interaction_scale=interaction_scale,
    )


def transform_inputs(
    expression: np.ndarray,
    age: np.ndarray,
    state: PreprocessingState,
) -> tuple[np.ndarray, np.ndarray]:
    x = (expression - state.expression_mean) / state.expression_scale
    z = (age - state.age_mean) / state.age_scale
    return x, z


def prepare_split(
    frame: pd.DataFrame,
    gene_names: tuple[str, ...],
    train_index: np.ndarray,
    validation_index: np.ndarray,
    config: DataConfig,
    permute_training_age: bool = False,
    permutation_seed: int | None = None,
    dtype: torch.dtype = torch.float64,
) -> PreparedSplit:
    train = frame.iloc[train_index].copy()
    validation = frame.iloc[validation_index].copy()
    train_age = train[config.age_column].to_numpy(dtype=float)
    if permute_training_age:
        rng = np.random.default_rng(permutation_seed)
        train_age = rng.permutation(train_age)

    x_train_raw = train[list(gene_names)].to_numpy(dtype=float)
    x_validation_raw = validation[list(gene_names)].to_numpy(dtype=float)
    state = fit_preprocessing(x_train_raw, train_age, config.eps)
    x_train, z_train = transform_inputs(x_train_raw, train_age, state)
    x_validation, z_validation = transform_inputs(
        x_validation_raw,
        validation[config.age_column].to_numpy(dtype=float),
        state,
    )

    tensor = lambda values: torch.as_tensor(values, dtype=dtype)
    return PreparedSplit(
        train_ids=train[config.sample_column].astype(str).to_numpy(),
        validation_ids=validation[config.sample_column].astype(str).to_numpy(),
        x_train=tensor(x_train),
        z_train=tensor(z_train),
        time_train=tensor(train[config.time_column].to_numpy(dtype=float)),
        event_train=tensor(train[config.event_column].to_numpy(dtype=float)),
        x_validation=tensor(x_validation),
        z_validation=tensor(z_validation),
        time_validation=tensor(validation[config.time_column].to_numpy(dtype=float)),
        event_validation=tensor(validation[config.event_column].to_numpy(dtype=float)),
        state=state,
        gene_names=gene_names,
    )


def train_single_split(
    split: PreparedSplit,
    run: int,
    seed: int,
    hidden_dim: int,
    gate_init: float,
    optimizer_config: OptimizerConfig,
    output_directory: str | Path,
) -> RunResult:
    set_seed(seed)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = f"run_{run:03d}"

    model = AGEDeepCox(
        n_genes=len(split.gene_names),
        hidden_dim=hidden_dim,
        gene_names=split.gene_names,
        gate_init=gate_init,
    )
    model.set_interaction_transform(
        torch.as_tensor(split.state.interaction_alpha),
        torch.as_tensor(split.state.interaction_scale),
    )
    fit = AGEDeepCoxOptimizer(model, optimizer_config).fit(
        split.x_train,
        split.z_train,
        split.time_train,
        split.event_train,
    )

    model.eval()
    with torch.no_grad():
        train_risk = model(split.x_train, split.z_train, False).total_risk.cpu().numpy()
        validation_risk = model(
            split.x_validation,
            split.z_validation,
            False,
        ).total_risk.cpu().numpy()

    checkpoint_path = output_directory / f"{stem}.pt"
    prediction_path = output_directory / f"{stem}_predictions.csv"
    coefficient_path = output_directory / f"{stem}_coefficients.csv"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "gene_names": split.gene_names,
            "hidden_dim": hidden_dim,
            "preprocessing": split.state,
            "fit_summary": asdict(fit),
            "seed": seed,
        },
        checkpoint_path,
    )
    predictions = pd.concat(
        [
            _prediction_frame(
                run, "train", split.train_ids, split.time_train, split.event_train, train_risk
            ),
            _prediction_frame(
                run,
                "validation",
                split.validation_ids,
                split.time_validation,
                split.event_validation,
                validation_risk,
            ),
        ],
        ignore_index=True,
    )
    predictions.to_csv(prediction_path, index=False)
    coefficients = pd.DataFrame(model.coefficient_table())
    coefficients.insert(0, "run", run)
    coefficients.to_csv(coefficient_path, index=False)
    return RunResult(
        run=run,
        seed=seed,
        iterations=fit.iterations,
        converged=fit.converged,
        objective=fit.objective,
        gate=fit.gate,
        checkpoint=str(checkpoint_path),
        predictions=str(prediction_path),
        coefficients=str(coefficient_path),
    )


def run_repeated_training(
    frame: pd.DataFrame,
    gene_names: tuple[str, ...],
    data_config: DataConfig,
    optimizer_config: OptimizerConfig,
    hidden_dim: int,
    gate_init: float,
    repeats: int,
    seed: int,
    output_directory: str | Path,
    permute_training_age: bool = False,
) -> pd.DataFrame:
    results: list[RunResult] = []
    for run in range(1, repeats + 1):
        run_seed = seed + run - 1
        train_index, validation_index = make_split(frame, data_config, run_seed)
        split = prepare_split(
            frame,
            gene_names,
            train_index,
            validation_index,
            data_config,
            permute_training_age=permute_training_age,
            permutation_seed=run_seed,
        )
        results.append(
            train_single_split(
                split,
                run,
                run_seed,
                hidden_dim,
                gate_init,
                optimizer_config,
                output_directory,
            )
        )
    table = pd.DataFrame(asdict(result) for result in results)
    output_directory = Path(output_directory)
    table.to_csv(output_directory / "runs.csv", index=False)
    pd.concat(
        [pd.read_csv(result.predictions) for result in results],
        ignore_index=True,
    ).to_csv(output_directory / "predictions.csv", index=False)
    pd.concat(
        [pd.read_csv(result.coefficients) for result in results],
        ignore_index=True,
    ).to_csv(output_directory / "coefficients.csv", index=False)
    return table


def summarize_selection_stability(
    coefficient_paths: list[str | Path],
    output_path: str | Path,
    tolerance: float = 1e-8,
) -> pd.DataFrame:
    if not coefficient_paths:
        raise ValueError("At least one coefficient file is required.")
    table = pd.concat(
        [pd.read_csv(path) for path in coefficient_paths],
        ignore_index=True,
    )
    required = {"run", "gene", "theta_reference", "theta_modification"}
    if not required.issubset(table.columns):
        raise ValueError(f"Coefficient files require columns: {sorted(required)}")
    table["reference_selected"] = table["theta_reference"].abs() > tolerance
    table["modification_selected"] = table["theta_modification"].abs() > tolerance
    stability = table.groupby("gene", as_index=False).agg(
        reference_frequency=("reference_selected", "mean"),
        modification_frequency=("modification_selected", "mean"),
        mean_theta_reference=("theta_reference", "mean"),
        mean_theta_modification=("theta_modification", "mean"),
        runs=("run", "nunique"),
    )
    stability["stable_reference"] = stability["reference_frequency"] > 0.5
    stability["stable_modification"] = stability["modification_frequency"] > 0.5
    stability.to_csv(output_path, index=False)
    return stability


def _prediction_frame(
    run: int,
    partition: str,
    sample_ids: np.ndarray,
    time: Tensor,
    event: Tensor,
    risk: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run": run,
            "partition": partition,
            "sample_id": sample_ids,
            "time": time.cpu().numpy(),
            "event": event.cpu().numpy().astype(int),
            "risk": risk,
        }
    )


def load_config(path: str | Path) -> dict[str, Any]:
    if yaml is None:
        raise ImportError("PyYAML is required to read configuration files.")
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train AGE-DeepCox.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("single", "repeated", "age-permutation"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--expression", required=True)
        subparser.add_argument("--clinical", required=True)
        subparser.add_argument("--config", required=True)
        subparser.add_argument("--output", required=True)
        subparser.add_argument("--seed", type=int)
        if command != "single":
            subparser.add_argument("--repeats", type=int)
    stability = subparsers.add_parser("stability")
    stability.add_argument("--coefficients", nargs="+", required=True)
    stability.add_argument("--output", required=True)
    stability.add_argument("--tolerance", type=float, default=1e-8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "stability":
        summarize_selection_stability(
            args.coefficients,
            args.output,
            args.tolerance,
        )
        return

    raw = load_config(args.config)
    data_config = DataConfig(**raw.get("data", {}))
    optimizer_config = OptimizerConfig(**raw.get("optimizer", {}))
    model_config = raw.get("model", {})
    experiment = raw.get("experiment", {})
    hidden_dim = int(model_config.get("hidden_dim", 16))
    gate_init = float(model_config.get("gate_init", 0.20))
    seed = int(args.seed if args.seed is not None else experiment.get("seed", 2027))
    repeats = int(
        args.repeats
        if getattr(args, "repeats", None) is not None
        else experiment.get("repeats", 50)
    )
    frame, _, gene_names = load_tables(
        args.expression,
        args.clinical,
        data_config,
    )
    if args.command == "single":
        train_index, validation_index = make_split(frame, data_config, seed)
        split = prepare_split(
            frame,
            gene_names,
            train_index,
            validation_index,
            data_config,
        )
        result = train_single_split(
            split,
            1,
            seed,
            hidden_dim,
            gate_init,
            optimizer_config,
            args.output,
        )
        Path(args.output, "runs.json").write_text(
            json.dumps(asdict(result), indent=2),
            encoding="utf-8",
        )
        return

    run_repeated_training(
        frame,
        gene_names,
        data_config,
        optimizer_config,
        hidden_dim,
        gate_init,
        repeats,
        seed,
        args.output,
        permute_training_age=args.command == "age-permutation",
    )


if __name__ == "__main__":
    main()


__all__ = [
    "DataConfig",
    "PreparedSplit",
    "PreprocessingState",
    "RunResult",
    "fit_preprocessing",
    "load_tables",
    "make_split",
    "prepare_split",
    "run_repeated_training",
    "summarize_selection_stability",
    "train_single_split",
    "transform_inputs",
]
