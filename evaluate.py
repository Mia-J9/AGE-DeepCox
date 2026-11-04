"""Evaluation utilities for AGE-DeepCox."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sksurv.util import Surv


DAYS_PER_YEAR = 365.25


@dataclass(frozen=True)
class EvaluationConfig:
    auc_years: tuple[float, ...] = (1.0, 3.0, 5.0)
    mean_auc_start_year: float = 1.0
    mean_auc_end_year: float = 5.0
    mean_auc_grid_size: int = 100

    def __post_init__(self) -> None:
        if not self.auc_years or any(year <= 0 for year in self.auc_years):
            raise ValueError("auc_years must contain positive values.")
        if self.mean_auc_start_year <= 0:
            raise ValueError("mean_auc_start_year must be positive.")
        if self.mean_auc_end_year <= self.mean_auc_start_year:
            raise ValueError("mean AUC end must exceed its start.")
        if self.mean_auc_grid_size < 2:
            raise ValueError("mean_auc_grid_size must be at least two.")


@dataclass(frozen=True)
class RunMetrics:
    run: str
    n_train: int
    n_validation: int
    validation_events: int
    c_index: float
    comparable_pairs: int
    concordant_pairs: int
    discordant_pairs: int
    tied_risk_pairs: int
    auc_1_year: float
    auc_3_year: float
    auc_5_year: float
    mean_auc_1_to_5_years: float


def harrell_c_index(
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
) -> tuple[float, int, int, int, int]:
    time, event, risk = _validate_arrays(time, event, risk)
    result = concordance_index_censored(event.astype(bool), time, risk)
    c_index, concordant, discordant, tied_risk, _ = result
    comparable = int(concordant + discordant + tied_risk)
    return (
        float(c_index),
        comparable,
        int(concordant),
        int(discordant),
        int(tied_risk),
    )


def time_dependent_auc(
    train_time: np.ndarray,
    train_event: np.ndarray,
    validation_time: np.ndarray,
    validation_event: np.ndarray,
    validation_risk: np.ndarray,
    config: EvaluationConfig,
) -> tuple[dict[float, float], float]:
    train_time, train_event, _ = _validate_arrays(
        train_time,
        train_event,
        np.zeros_like(train_time, dtype=float),
    )
    validation_time, validation_event, validation_risk = _validate_arrays(
        validation_time,
        validation_event,
        validation_risk,
    )
    survival_train = Surv.from_arrays(train_event.astype(bool), train_time)
    survival_validation = Surv.from_arrays(
        validation_event.astype(bool), validation_time
    )

    point_times = np.asarray(config.auc_years, dtype=float) * DAYS_PER_YEAR
    mean_times = np.linspace(
        config.mean_auc_start_year * DAYS_PER_YEAR,
        config.mean_auc_end_year * DAYS_PER_YEAR,
        config.mean_auc_grid_size,
    )
    evaluation_times = np.unique(np.concatenate([point_times, mean_times]))
    _validate_auc_times(evaluation_times, train_time, validation_time)

    auc_values, _ = cumulative_dynamic_auc(
        survival_train,
        survival_validation,
        validation_risk,
        evaluation_times,
    )
    auc_by_time = dict(zip(evaluation_times, auc_values))
    point_auc = {
        year: float(auc_by_time[year * DAYS_PER_YEAR])
        for year in config.auc_years
    }

    mean_mask = (evaluation_times >= mean_times[0]) & (
        evaluation_times <= mean_times[-1]
    )
    _, mean_auc = cumulative_dynamic_auc(
        survival_train,
        survival_validation,
        validation_risk,
        evaluation_times[mean_mask],
    )
    return point_auc, float(mean_auc)


def evaluate_run(
    run: str,
    training: pd.DataFrame,
    validation: pd.DataFrame,
    config: EvaluationConfig,
    time_column: str = "time",
    event_column: str = "event",
    risk_column: str = "risk",
) -> RunMetrics:
    required = {time_column, event_column}
    missing_train = required - set(training.columns)
    missing_validation = (required | {risk_column}) - set(validation.columns)
    if missing_train:
        raise ValueError(f"Training rows are missing columns: {sorted(missing_train)}")
    if missing_validation:
        raise ValueError(
            f"Validation rows are missing columns: {sorted(missing_validation)}"
        )

    train_time = training[time_column].to_numpy(float)
    train_event = training[event_column].to_numpy(int)
    validation_time = validation[time_column].to_numpy(float)
    validation_event = validation[event_column].to_numpy(int)
    validation_risk = validation[risk_column].to_numpy(float)

    c_index, comparable, concordant, discordant, tied_risk = harrell_c_index(
        validation_time,
        validation_event,
        validation_risk,
    )
    point_auc, mean_auc = time_dependent_auc(
        train_time,
        train_event,
        validation_time,
        validation_event,
        validation_risk,
        config,
    )
    required_years = {1.0, 3.0, 5.0}
    if not required_years.issubset(point_auc):
        raise ValueError("auc_years must include 1, 3, and 5 years.")

    return RunMetrics(
        run=str(run),
        n_train=len(training),
        n_validation=len(validation),
        validation_events=int(validation_event.sum()),
        c_index=c_index,
        comparable_pairs=comparable,
        concordant_pairs=concordant,
        discordant_pairs=discordant,
        tied_risk_pairs=tied_risk,
        auc_1_year=point_auc[1.0],
        auc_3_year=point_auc[3.0],
        auc_5_year=point_auc[5.0],
        mean_auc_1_to_5_years=mean_auc,
    )


def evaluate_predictions(
    predictions: pd.DataFrame,
    config: EvaluationConfig,
    run_column: str = "run",
    partition_column: str = "partition",
    time_column: str = "time",
    event_column: str = "event",
    risk_column: str = "risk",
) -> pd.DataFrame:
    required = {run_column, partition_column, time_column, event_column}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")

    rows = []
    for run, frame in predictions.groupby(run_column, sort=True):
        partition = frame[partition_column].astype(str).str.lower()
        training = frame.loc[partition.eq("train")]
        validation = frame.loc[partition.isin(["validation", "val"])]
        if training.empty or validation.empty:
            raise ValueError(f"Run {run} must contain train and validation rows.")
        rows.append(
            asdict(
                evaluate_run(
                    str(run),
                    training,
                    validation,
                    config,
                    time_column,
                    event_column,
                    risk_column,
                )
            )
        )
    if not rows:
        raise ValueError("No runs were found in the prediction table.")
    return pd.DataFrame(rows)


def summarize_runs(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "c_index",
        "auc_1_year",
        "auc_3_year",
        "auc_5_year",
        "mean_auc_1_to_5_years",
    ]
    missing = set(metric_columns) - set(metrics.columns)
    if missing:
        raise ValueError(f"Metrics table is missing columns: {sorted(missing)}")

    rows = []
    for column in metric_columns:
        values = pd.to_numeric(metrics[column], errors="raise")
        rows.append(
            {
                "metric": column,
                "n_runs": int(values.notna().sum()),
                "mean": float(values.mean()),
                "standard_deviation": float(values.std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def _validate_arrays(
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event)
    risk = np.asarray(risk, dtype=float)
    if time.ndim != 1 or event.ndim != 1 or risk.ndim != 1:
        raise ValueError("time, event, and risk must be one-dimensional.")
    if not (time.shape == event.shape == risk.shape):
        raise ValueError("time, event, and risk must have identical shapes.")
    if len(time) == 0:
        raise ValueError("Survival arrays cannot be empty.")
    if not np.isfinite(time).all() or not np.isfinite(risk).all():
        raise ValueError("time and risk must be finite.")
    if np.any(time <= 0):
        raise ValueError("Survival times must be positive.")
    if not np.isin(event, [0, 1]).all():
        raise ValueError("event must be coded as 0/1.")
    if not np.any(event == 1):
        raise ValueError("At least one observed event is required.")
    return time, event.astype(int), risk


def _validate_auc_times(
    evaluation_times: np.ndarray,
    train_time: np.ndarray,
    validation_time: np.ndarray,
) -> None:
    lower = float(validation_time.min())
    upper = float(validation_time.max())
    if np.any(evaluation_times < lower) or np.any(evaluation_times >= upper):
        raise ValueError(
            "AUC evaluation times must lie within the validation follow-up "
            f"interval [{lower:.3f}, {upper:.3f})."
        )
    if float(train_time.max()) <= float(evaluation_times.max()):
        raise ValueError(
            "Training follow-up must extend beyond the largest AUC time for "
            "IPCW estimation."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-column", default="run")
    parser.add_argument("--partition-column", default="partition")
    parser.add_argument("--time-column", default="time")
    parser.add_argument("--event-column", default="event")
    parser.add_argument("--risk-column", default="risk")
    parser.add_argument("--mean-auc-grid-size", default=100, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    config = EvaluationConfig(mean_auc_grid_size=args.mean_auc_grid_size)
    metrics = evaluate_predictions(
        predictions,
        config,
        args.run_column,
        args.partition_column,
        args.time_column,
        args.event_column,
        args.risk_column,
    )
    summary = summarize_runs(metrics)
    args.output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output / "metrics_by_run.csv", index=False)
    summary.to_csv(args.output / "metrics_summary.csv", index=False)


if __name__ == "__main__":
    main()
