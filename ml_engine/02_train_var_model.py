from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

from data_pipeline import handle_outliers

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "shockwave_training_data.csv"
MODEL_PATH = BASE_DIR / "shockwave_var_model.pkl"
BACKTEST_REPORT_PATH = BASE_DIR / "shockwave_backtest_report.json"
BACKTEST_PREDICTIONS_PATH = BASE_DIR / "shockwave_backtest_predictions.csv"

LEADING_INDICATORS = ["eia_brent_price", "eia_natural_gas_price"]
TARGET_VARIABLES = ["doeb_import_volume", "doeb_diesel_sales"]
BACKTEST_STEPS = int(os.getenv("BACKTEST_STEPS", "3"))
MAX_GRANGER_LAG = int(os.getenv("MAX_GRANGER_LAG", "6"))
MAX_VAR_LAG = int(os.getenv("MAX_VAR_LAG", "12"))
MIN_TRAINING_ROWS = int(os.getenv("MIN_TRAINING_ROWS", "36"))


def adf_pvalue(series: pd.Series) -> float:
    return adfuller(series.dropna(), autolag="AIC")[1]


def prepare_stationary_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, bool, dict[str, float]]:
    pvalues = {column: adf_pvalue(df[column]) for column in df.columns}
    logger.info("ADF p-values: %s", pvalues)

    needs_difference = any(pvalue > 0.05 for pvalue in pvalues.values())
    if not needs_difference:
        return df.copy(), False, pvalues

    stationary_df = df.diff().dropna()
    logger.info("Applied first differencing because at least one series failed stationarity.")
    return stationary_df, True, pvalues


def scale_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Scale dataframe to zero mean and unit variance."""
    means = df.mean()
    stds = df.std(ddof=0).replace(0, 1.0)
    scaled_df = (df - means) / stds
    return scaled_df, means, stds


def run_pairwise_granger_tests(df: pd.DataFrame, max_lag: int = 6) -> pd.DataFrame:
    safe_max_lag = min(max_lag, max(1, len(df) // 5))
    all_rows: list[dict[str, Any]] = []

    for leading_indicator in LEADING_INDICATORS:
        for target_variable in TARGET_VARIABLES:
            test_input = df[[target_variable, leading_indicator]]
            results = grangercausalitytests(test_input, maxlag=safe_max_lag, verbose=False)

            for lag, test_result in results.items():
                all_rows.append(
                    {
                        "leading_indicator": leading_indicator,
                        "target_variable": target_variable,
                        "lag": lag,
                        "ssr_ftest_pvalue": test_result[0]["ssr_ftest"][1],
                        "lrtest_pvalue": test_result[0]["lrtest"][1],
                    }
                )

    summary_df = pd.DataFrame(all_rows)
    logger.info("Granger causality summary:\n%s", summary_df.to_string(index=False))
    return summary_df


def select_var_lag(model: VAR, max_var_lag: int) -> tuple[int, Any]:
    safe_max_var_lag = min(max_var_lag, max(2, len(model.endog) // 4))
    lag_selection = model.select_order(maxlags=safe_max_var_lag)

    # Voting mechanism across criteria to avoid overfitting
    votes = [lag_selection.aic, lag_selection.bic, lag_selection.hqic, lag_selection.fpe]
    valid_votes = [v for v in votes if v is not None and v > 0]
    
    if not valid_votes:
        raise ValueError("Unable to determine VAR lag order from criteria.")
    
    # Select the most frequently voted lag, fallback to minimum if tie
    from collections import Counter
    vote_counts = Counter(valid_votes)
    selected_lag = max(vote_counts, key=vote_counts.get)

    return max(1, int(selected_lag)), lag_selection


def train_var_model(
    df: pd.DataFrame,
    *,
    max_var_lag: int = MAX_VAR_LAG,
    print_summary: bool = True,
) -> tuple[Any, int, dict[str, Any]]:
    model = VAR(df)
    selected_lag, lag_selection = select_var_lag(model, max_var_lag=max_var_lag)
    logger.info("Selected VAR lag order: %s", selected_lag)

    fitted_model = model.fit(selected_lag)
    irf_periods = 24
    irf = fitted_model.irf(irf_periods)
    lag_metadata = {
        "selected_lag": selected_lag,
        "aic": lag_selection.aic,
        "bic": lag_selection.bic,
        "hqic": lag_selection.hqic,
        "fpe": lag_selection.fpe,
        "irf_periods": irf_periods,
    }
    if print_summary:
        logger.info("\n%s", fitted_model.summary())
    return fitted_model, selected_lag, {"lag_metadata": lag_metadata, "irf": irf}


def reconstruct_level_forecast(
    forecast_row: np.ndarray,
    *,
    train_columns: list[str],
    is_differenced: bool,
    train_raw_df: pd.DataFrame,
) -> pd.Series:
    level_forecast = pd.Series(forecast_row, index=train_columns, dtype=float)
    if is_differenced:
        level_forecast = level_forecast.add(train_raw_df[train_columns].iloc[-1], fill_value=0.0)
    return level_forecast


def safe_mape(actual: pd.Series, predicted: pd.Series) -> float:
    denominator = actual.replace(0, np.nan).abs()
    percentage_error = ((actual - predicted).abs() / denominator).dropna()
    if percentage_error.empty:
        return float("nan")
    return float(percentage_error.mean() * 100.0)


def directional_accuracy(actual: pd.Series, predicted: pd.Series, previous_actual: pd.Series) -> float:
    """Calculate Mean Directional Accuracy (MDA)."""
    actual_direction = np.sign(actual - previous_actual)
    predicted_direction = np.sign(predicted - previous_actual)
    correct_directions = (actual_direction == predicted_direction).sum()
    total = len(actual)
    if total == 0:
        return float("nan")
    return float(correct_directions / total * 100.0)


def build_backtest_report(
    raw_df: pd.DataFrame,
    *,
    backtest_steps: int = BACKTEST_STEPS,
    min_training_rows: int = MIN_TRAINING_ROWS,
    max_var_lag: int = MAX_VAR_LAG,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if len(raw_df) < (min_training_rows + backtest_steps):
        raise ValueError(
            "Not enough rows to run the requested backtest. "
            f"Need at least {min_training_rows + backtest_steps} rows, got {len(raw_df)}."
        )

    columns = list(raw_df.columns)
    prediction_rows: list[dict[str, Any]] = []

    start_idx = len(raw_df) - backtest_steps
    for test_idx in range(start_idx, len(raw_df)):
        train_raw = raw_df.iloc[:test_idx].copy()
        actual_row = raw_df.iloc[test_idx].copy()

        stationary_train, is_differenced, _ = prepare_stationary_dataframe(train_raw)
        if len(stationary_train) < max(8, min_training_rows // 2):
            raise ValueError(
                "Training window became too small after stationarity preprocessing. "
                f"Rows available: {len(stationary_train)}."
            )

        scaled_train, train_means, train_stds = scale_dataframe(stationary_train)

        var_model, selected_lag, _ = train_var_model(
            scaled_train,
            max_var_lag=max_var_lag,
            print_summary=False,
        )
        history_matrix = np.asarray(var_model.endog[-selected_lag:], dtype=float).copy()
        next_forecast = var_model.forecast(history_matrix, steps=1)[0]
        level_forecast = reconstruct_level_forecast(
            next_forecast,
            train_columns=columns,
            is_differenced=is_differenced,
            train_raw_df=train_raw,
            means=train_means,
            stds=train_stds,
        )

        row: dict[str, Any] = {
            "month": raw_df.index[test_idx].date().isoformat(),
            "train_end": train_raw.index[-1].date().isoformat(),
            "selected_lag": selected_lag,
            "is_differenced": is_differenced,
        }
        for column in columns:
            row[f"actual_{column}"] = float(actual_row[column])
            row[f"predicted_{column}"] = float(level_forecast[column])
            row[f"abs_error_{column}"] = float(abs(actual_row[column] - level_forecast[column]))
        prediction_rows.append(row)

    predictions_df = pd.DataFrame(prediction_rows)

    metrics: dict[str, dict[str, float | None]] = {}
    for column in TARGET_VARIABLES:
        actual = predictions_df[f"actual_{column}"]
        predicted = predictions_df[f"predicted_{column}"]
        
        # Calculate directional accuracy
        start_idx = len(raw_df) - backtest_steps
        previous_actual_list = []
        for i in range(start_idx, len(raw_df)):
            previous_actual_list.append(raw_df[column].iloc[i - 1])
        previous_actual = pd.Series(previous_actual_list, index=actual.index)
        mda = directional_accuracy(actual, predicted, previous_actual)
        
        # Calculate R-squared
        ss_res = ((actual - predicted) ** 2).sum()
        ss_tot = ((actual - actual.mean()) ** 2).sum()
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else float("nan")

        errors = actual - predicted
        rmse = math.sqrt(float((errors.pow(2)).mean()))
        metrics[column] = {
            "mae": round(float(errors.abs().mean()), 4),
            "rmse": round(rmse, 4),
            "mape": round(safe_mape(actual, predicted), 4),
            "mda": round(mda, 4),
            "r_squared": round(float(r_squared), 4),
        }

    report = {
        "backtest_steps": backtest_steps,
        "min_training_rows": min_training_rows,
        "targets": TARGET_VARIABLES,
        "test_window_start": predictions_df["month"].iloc[0],
        "test_window_end": predictions_df["month"].iloc[-1],
        "metrics": metrics,
    }
    return report, predictions_df


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found at {DATA_PATH}. Run 01b_generate_mock_doeb_data.py first."
        )

    raw_df = pd.read_csv(DATA_PATH, parse_dates=["Date"], index_col="Date").asfreq("MS")
    if len(raw_df) < MIN_TRAINING_ROWS:
        raise ValueError(
            f"Training data must contain at least {MIN_TRAINING_ROWS} rows, got {len(raw_df)}."
        )
    
    raw_df = handle_outliers(raw_df)

    backtest_report, backtest_predictions = build_backtest_report(
        raw_df,
        backtest_steps=BACKTEST_STEPS,
        min_training_rows=MIN_TRAINING_ROWS,
        max_var_lag=MAX_VAR_LAG,
    )
    backtest_predictions.to_csv(BACKTEST_PREDICTIONS_PATH, index=False)
    logger.info("Backtest summary:\n%s", pd.DataFrame(backtest_report["metrics"]).T.to_string())

    stationary_df, is_differenced, adf_summary = prepare_stationary_dataframe(raw_df)
    scaled_df, train_means, train_stds = scale_dataframe(stationary_df)

    granger_summary = run_pairwise_granger_tests(scaled_df, max_lag=MAX_GRANGER_LAG)
    var_model, selected_lag, model_extras = train_var_model(
        scaled_df,
        max_var_lag=MAX_VAR_LAG,
    )

    leading_relationships = (
        granger_summary[granger_summary["ssr_ftest_pvalue"] < 0.05]
        .sort_values(["target_variable", "ssr_ftest_pvalue", "lag"])
        .to_dict(orient="records")
    )

    model_bundle = {
        "var_results": var_model,
        "selected_lag": selected_lag,
        "is_differenced": is_differenced,
        "train_columns": [str(c) for c in stationary_df.columns],
        "train_end": str(stationary_df.index.max()),
        "adf_summary": {str(k): float(v) for k, v in adf_summary.items()},
        "granger_summary": granger_summary.astype(str).to_dict(orient="records"),
        "leading_relationships": leading_relationships,
        "lag_metadata": model_extras["lag_metadata"],
        "backtest_report": backtest_report,
        "train_means": {str(k): float(v) for k, v in train_means.to_dict().items()},
        "train_std": {str(k): float(v) for k, v in train_stds.to_dict().items()},
        "model_status": "ready",
    }
    joblib.dump(model_bundle, MODEL_PATH)
    with BACKTEST_REPORT_PATH.open("w", encoding="utf-8") as report_file:
        json.dump(backtest_report, report_file, indent=2)

    logger.info("Saved VAR model bundle to %s", MODEL_PATH)
    logger.info("Saved backtest report to %s", BACKTEST_REPORT_PATH)
    logger.info("Saved backtest predictions to %s", BACKTEST_PREDICTIONS_PATH)


if __name__ == "__main__":
    main()
n__":
    main()
