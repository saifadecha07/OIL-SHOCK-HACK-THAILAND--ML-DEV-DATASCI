from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from data_pipeline import build_real_local_targets, build_synthetic_local_targets, handle_outliers


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "shockwave_training_data.csv"
EIA_API_KEY = os.getenv("EIA_API_KEY", "YOUR_EIA_API_KEY")
DOEB_IMPORT_PATH = os.getenv("DOEB_IMPORT_PATH")
DOEB_DIESEL_PATH = os.getenv("DOEB_DIESEL_PATH")
EPPO_MONTHLY_PATH = os.getenv("EPPO_MONTHLY_PATH")


@dataclass
class EIADataFetcher:
    """Fetch EIA API v2 data with retry logic suitable for notebook/script pipelines."""

    api_key: str
    base_url: str = "https://api.eia.gov/v2"
    timeout: int = 30
    max_retries: int = 5
    backoff_factor: float = 1.5
    session: requests.Session = field(default_factory=requests.Session)

    def _request(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        request_params = {"api_key": self.api_key, **params}

        for attempt in range(self.max_retries):
            response = self.session.get(url, params=request_params, timeout=self.timeout)
            if response.status_code == 429:
                wait_seconds = self.backoff_factor ** attempt
                print(f"Rate limited by EIA API. Retrying in {wait_seconds:.1f}s...")
                time.sleep(wait_seconds)
                continue

            response.raise_for_status()
            payload = response.json()
            if "response" not in payload:
                raise ValueError(f"Unexpected EIA response shape: {payload}")
            return payload

        raise RuntimeError("EIA API retries exhausted after repeated rate-limit responses.")

    def fetch_monthly_series(
        self,
        *,
        series_id: str,
        value_name: str,
        start: str = "2015-01",
        end: str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch a monthly EIA series through the API v2 seriesid route.

        Expected examples:
        - PET.RBRTE.M for Brent spot price
        - NG.RNGWHHD.M for Henry Hub natural gas price
        """

        endpoint = f"seriesid/{series_id}"
        params: dict[str, Any] = {"out": "json"}
        if start is not None:
            params["start"] = start
        if end is not None:
            params["end"] = end

        payload = self._request(endpoint, params)
        rows = payload["response"].get("data", [])
        if not rows:
            raise ValueError(f"No rows returned for EIA series {series_id}.")

        df = pd.DataFrame(rows)
        period_column = "period" if "period" in df.columns else "date"
        value_column = "value" if "value" in df.columns else df.columns[-1]

        df["Date"] = pd.to_datetime(df[period_column], errors="coerce")
        df[value_name] = pd.to_numeric(df[value_column], errors="coerce")

        result = (
            df[["Date", value_name]]
            .dropna()
            .sort_values("Date")
            .drop_duplicates(subset=["Date"])
            .set_index("Date")
            .asfreq("MS")
        )
        return result


def build_global_signal_frame(fetcher: EIADataFetcher) -> pd.DataFrame:
    brent_df = fetcher.fetch_monthly_series(
        series_id="PET.RBRTE.M",
        value_name="eia_brent_price",
    )
    natgas_df = fetcher.fetch_monthly_series(
        series_id="NG.RNGWHHD.M",
        value_name="eia_natural_gas_price",
    )
    return brent_df.join(natgas_df, how="inner").sort_index()


def main() -> None:
    if EIA_API_KEY == "YOUR_EIA_API_KEY":
        raise ValueError(
            "Missing EIA API key. Set EIA_API_KEY in your environment or replace "
            "the placeholder in this script before running."
        )

    fetcher = EIADataFetcher(api_key=EIA_API_KEY)
    global_df = build_global_signal_frame(fetcher)

    local_df, source_map = build_real_local_targets(
        doeb_import_path=DOEB_IMPORT_PATH,
        doeb_diesel_path=DOEB_DIESEL_PATH,
        eppo_monthly_path=EPPO_MONTHLY_PATH,
    )
    if local_df is None:
        print("Real DOEB/EPPO target files not found. Falling back to synthetic local targets.")
        local_df = build_synthetic_local_targets(global_df, lag_months=3, seed=42)
        source_map["target_mode"] = "synthetic_fallback"
    else:
        source_map["target_mode"] = "official_local_files"

    training_df = global_df.join(local_df, how="inner").sort_index().dropna()
    training_df = handle_outliers(training_df)
    training_df.index.name = "Date"
    training_df.to_csv(OUTPUT_PATH)

    print(f"Saved training data to {OUTPUT_PATH}")
    print(f"Data sources: {source_map}")
    print(training_df.tail(12).to_string())


if __name__ == "__main__":
    main()
