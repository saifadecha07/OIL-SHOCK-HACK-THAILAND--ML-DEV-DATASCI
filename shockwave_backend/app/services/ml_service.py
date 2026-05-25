from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.core.config import settings


class VARModelService:
    """
    Service layer responsible for loading the trained VAR artifact and running
    scenario simulations against it.

    The model artifact is expected to be the joblib bundle created in the ML
    training phase. If the model was trained on differenced data, this service
    reconstructs level forecasts using the last observed row from the training
    CSV.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        training_data_path: Path | None = None,
    ) -> None:
        project_root = Path(__file__).resolve().parents[3]
        self.model_path = model_path or self._resolve_existing_path(
            project_root / "ml_engine" / "artifacts" / "models" / "shockwave_var_model.pkl",
            project_root / "ml_engine" / "shockwave_var_model.pkl",
        )
        self.training_data_path = training_data_path or self._resolve_existing_path(
            project_root / "ml_engine" / "data" / "processed" / "shockwave_modeling_frame.csv",
            project_root / "ml_engine" / "shockwave_training_data.csv",
        )
        self.use_mock_model = False

        self.eia_column = "eia_brent_price"
        self.natgas_column = "eia_natural_gas_price"
        self.import_column = "doeb_import_volume"
        self.diesel_column = "doeb_diesel_sales"

        if not self.model_path.exists():
            if not settings.allow_mock_model:
                raise FileNotFoundError(
                    f"VAR model artifact not found at {self.model_path}. "
                    "Train the model before starting the API."
                )

            self._configure_mock_mode()
            return

        self.model_bundle: dict[str, Any] = joblib.load(self.model_path)
        self.var_results = self.model_bundle["var_results"]
        self.selected_lag = int(self.model_bundle.get("selected_lag", self.var_results.k_ar))
        self.is_differenced = bool(self.model_bundle.get("is_differenced", False))
        self.train_columns = list(
            self.model_bundle.get("train_columns", getattr(self.var_results, "names", []))
        )
        self.train_means = {
            key: float(value) for key, value in self.model_bundle.get("train_means", {}).items()
        }
        self.train_std = {
            key: float(value) for key, value in self.model_bundle.get("train_std", {}).items()
        }
        self.leading_indicators = [
            column for column in ("eia_brent_price", "eia_natural_gas_price") if column in self.train_columns
        ]
        self.target_variables = [
            column for column in ("doeb_import_volume", "doeb_diesel_sales") if column in self.train_columns
        ]
        self.raw_history = self._load_raw_history()

        missing_columns = {
            column
            for column in (self.eia_column, self.import_column, self.diesel_column)
            if column not in self.train_columns
        }
        if missing_columns:
            raise ValueError(
                "The VAR artifact is missing expected variables: "
                f"{', '.join(sorted(missing_columns))}"
            )

    @staticmethod
    def _resolve_existing_path(primary: Path, fallback: Path) -> Path:
        return primary if primary.exists() else fallback

    def _configure_mock_mode(self) -> None:
        self.use_mock_model = True
        self.model_bundle = {
            "model_status": "mock_ready",
            "leading_relationships": [],
            "lag_metadata": {"mode": "mock", "reason": "artifact_missing"},
        }
        self.var_results = None
        self.selected_lag = 3
        self.is_differenced = False
        self.train_columns = [
            self.eia_column,
            self.natgas_column,
            self.import_column,
            self.diesel_column,
        ]
        self.train_means = {column: 0.0 for column in self.train_columns}
        self.train_std = {column: 1.0 for column in self.train_columns}
        self.leading_indicators = [self.eia_column, self.natgas_column]
        self.target_variables = [self.import_column, self.diesel_column]
        self.raw_history = None

    def _simulate_mock_shock(self, shock_percentage: float, steps: int) -> dict[str, Any]:
        forecast_index = self._build_forecast_index(steps)
        baseline_import = 90.0
        baseline_diesel = 66.0
        import_drop_scale = 0.11
        diesel_drop_scale = 0.07
        lag_curve = np.array([1 - np.exp(-(idx + 1) / 2.8) for idx in range(steps)])

        results: list[dict[str, Any]] = []
        for idx, timestamp in enumerate(forecast_index):
            seasonality = np.sin((idx + 1) * np.pi / 6)
            baseline_import_volume = baseline_import + (1.8 * seasonality) + (0.35 * idx)
            baseline_diesel_sales = baseline_diesel + (1.3 * seasonality) + (0.22 * idx)

            import_delta = -(shock_percentage * import_drop_scale * lag_curve[idx])
            diesel_delta = -(shock_percentage * diesel_drop_scale * lag_curve[idx])

            results.append(
                {
                    "month": timestamp.date(),
                    "baseline_doeb_import_volume": round(float(baseline_import_volume), 4),
                    "shocked_doeb_import_volume": round(float(baseline_import_volume + import_delta), 4),
                    "delta_doeb_import_volume": round(float(import_delta), 4),
                    "baseline_doeb_diesel_sales": round(float(baseline_diesel_sales), 4),
                    "shocked_doeb_diesel_sales": round(float(baseline_diesel_sales + diesel_delta), 4),
                    "delta_doeb_diesel_sales": round(float(diesel_delta), 4),
                }
            )

        forecast_df = pd.DataFrame(results)
        critical_row = forecast_df.loc[forecast_df["delta_doeb_import_volume"].idxmin()]
        return {
            "leading_indicators": self.leading_indicators,
            "summary": {
                "critical_month": critical_row["month"],
                "max_drop_import_volume": round(float(forecast_df["delta_doeb_import_volume"].min()), 4),
                "max_drop_diesel_sales": round(float(forecast_df["delta_doeb_diesel_sales"].min()), 4),
                "shock_transmission_method": "Mock lag simulator",
                "selected_lag_months": self.selected_lag,
            },
            "forecasts": results,
        }

    def _load_raw_history(self) -> pd.DataFrame | None:
        if not self.training_data_path.exists():
            return None

        history = pd.read_csv(
            self.training_data_path,
            parse_dates=["Date"],
            index_col="Date",
        ).asfreq("MS")
        return history

    def _forecast_levels(self, model_scale_forecast: np.ndarray) -> pd.DataFrame:
        forecast_index = self._build_forecast_index(len(model_scale_forecast))
        forecast_df = pd.DataFrame(
            model_scale_forecast,
            index=forecast_index,
            columns=self.train_columns,
        )

        # Reverse StandardScaler: forecast is in scaled space (model space), convert to raw units.
        means_series = pd.Series({col: self.train_means.get(col, 0.0) for col in self.train_columns})
        stds_series = pd.Series({col: self.train_std.get(col, 1.0) for col in self.train_columns})
        forecast_df = forecast_df.multiply(stds_series).add(means_series)

        # When the model was trained on differences, statsmodels forecasts future
        # deltas (already unscaled above). Cumsum + last observed level = level forecast.
        if self.is_differenced:
            if self.raw_history is None:
                raise ValueError(
                    "The VAR model was trained on differenced data, but the raw "
                    "training history is unavailable for level reconstruction."
                )

            last_raw_row = self.raw_history[self.train_columns].iloc[-1]
            forecast_df = forecast_df.cumsum().add(last_raw_row, axis="columns")

        return forecast_df

    def _build_forecast_index(self, steps: int) -> pd.DatetimeIndex:
        if self.raw_history is not None and not self.raw_history.empty:
            start = self.raw_history.index[-1] + pd.offsets.MonthBegin()
        else:
            train_end = pd.Timestamp(self.model_bundle.get("train_end", pd.Timestamp.utcnow()))
            start = train_end + pd.offsets.MonthBegin()

        return pd.date_range(start=start, periods=steps, freq="MS")

    def _build_baseline_model_forecast(self, steps: int) -> pd.DataFrame:
        history_matrix = np.asarray(self.var_results.endog[-self.selected_lag :], dtype=float).copy()
        baseline_forecast = self.var_results.forecast(history_matrix, steps=steps)
        return pd.DataFrame(baseline_forecast, columns=self.train_columns)

    def _build_irf_shock_adjustment(self, shock_percentage: float, steps: int) -> pd.DataFrame:
        """
        Build an IRF-based delta path for the shock.

        We convert the requested percentage shock into model units, then scale the
        impulse responses from the EIA variables across the forecast horizon. Brent
        gets the full requested shock; natural gas receives a smaller coupled shock
        to mimic spillover across global fuel markets.
        """

        irf = self.var_results.irf(steps)
        irf_array = irf.irfs[1 : steps + 1]
        response_frame = pd.DataFrame(0.0, index=range(steps), columns=self.train_columns)

        impulse_scalars: dict[str, float] = {}
        if self.raw_history is not None:
            last_eia = float(self.raw_history[self.eia_column].iloc[-1])
            impulse_scalars[self.eia_column] = last_eia * (shock_percentage / 100.0)

            if self.natgas_column in self.raw_history.columns:
                last_natgas = float(self.raw_history[self.natgas_column].iloc[-1])
                impulse_scalars[self.natgas_column] = last_natgas * ((shock_percentage * 0.35) / 100.0)

        for impulse_column, raw_delta in impulse_scalars.items():
            if impulse_column not in self.train_columns:
                continue

            impulse_idx = self.train_columns.index(impulse_column)
            model_scale = self.train_std.get(impulse_column, 1.0)
            scaled_delta = raw_delta / model_scale

            for response_idx, response_column in enumerate(self.train_columns):
                response_frame[response_column] += irf_array[:, response_idx, impulse_idx] * scaled_delta

        return response_frame

    def simulate_shock(self, shock_percentage: float, steps: int) -> dict[str, Any]:
        if steps < 1:
            raise ValueError("Forecast steps must be at least 1.")
        if self.use_mock_model:
            return self._simulate_mock_shock(shock_percentage, steps)

        baseline_forecast = self._build_baseline_model_forecast(steps)
        irf_adjustment = self._build_irf_shock_adjustment(shock_percentage, steps)
        shocked_forecast = baseline_forecast.add(irf_adjustment, fill_value=0.0)

        baseline_levels = self._forecast_levels(baseline_forecast.to_numpy())
        shocked_levels = self._forecast_levels(shocked_forecast.to_numpy())

        results: list[dict[str, Any]] = []
        for timestamp in baseline_levels.index:
            baseline_row = baseline_levels.loc[timestamp]
            shocked_row = shocked_levels.loc[timestamp]
            results.append(
                {
                    "month": timestamp.date(),
                    "baseline_doeb_import_volume": round(float(baseline_row[self.import_column]), 4),
                    "shocked_doeb_import_volume": round(float(shocked_row[self.import_column]), 4),
                    "delta_doeb_import_volume": round(
                        float(shocked_row[self.import_column] - baseline_row[self.import_column]),
                        4,
                    ),
                    "baseline_doeb_diesel_sales": round(float(baseline_row[self.diesel_column]), 4),
                    "shocked_doeb_diesel_sales": round(float(shocked_row[self.diesel_column]), 4),
                    "delta_doeb_diesel_sales": round(
                        float(shocked_row[self.diesel_column] - baseline_row[self.diesel_column]),
                        4,
                    ),
                }
            )

        forecast_df = pd.DataFrame(results)
        critical_row = forecast_df.loc[forecast_df["delta_doeb_import_volume"].idxmin()]
        summary = {
            "critical_month": critical_row["month"],
            "max_drop_import_volume": round(float(forecast_df["delta_doeb_import_volume"].min()), 4),
            "max_drop_diesel_sales": round(float(forecast_df["delta_doeb_diesel_sales"].min()), 4),
            "shock_transmission_method": "Impulse Response Function (IRF)",
            "selected_lag_months": self.selected_lag,
        }

        return {
            "leading_indicators": self.leading_indicators,
            "summary": summary,
            "forecasts": results,
        }

    def get_health_status(self) -> dict[str, Any]:
        return {
            "status": self.model_bundle.get("model_status", "ready"),
            "selected_lag_months": self.selected_lag,
            "leading_indicators": self.leading_indicators,
            "target_variables": self.target_variables,
            "is_differenced": self.is_differenced,
            "mock_mode": self.use_mock_model,
        }

    def get_model_metadata(self) -> dict[str, Any]:
        granger_summary = self.model_bundle.get("leading_relationships", [])
        lag_metadata = self.model_bundle.get("lag_metadata", {})
        return {
            "selected_lag_months": self.selected_lag,
            "shock_transmission_method": (
                "Mock lag simulator" if self.use_mock_model else "Impulse Response Function (IRF)"
            ),
            "leading_indicators": self.leading_indicators,
            "target_variables": self.target_variables,
            "granger_relationships": granger_summary,
            "lag_metadata": lag_metadata,
        }


@lru_cache
def get_var_model_service() -> VARModelService:
    return VARModelService()
