from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def handle_outliers(df: pd.DataFrame, threshold: float = 1.5) -> pd.DataFrame:
    """
    Cap outliers using Tukey IQR fences (median/IQR are robust to the outliers
    being detected, unlike mean/std which are distorted by the same extremes).
    threshold=1.5 is the standard Tukey inner fence.
    """
    df_cleaned = df.copy()
    for col in df_cleaned.columns:
        if pd.api.types.is_numeric_dtype(df_cleaned[col]):
            q1 = df_cleaned[col].quantile(0.25)
            q3 = df_cleaned[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                df_cleaned[col] = df_cleaned[col].clip(lower=lower_bound, upper=upper_bound)
    return df_cleaned


def load_timeseries_csv(
    file_path: str | Path,
    *,
    date_column: str,
    value_column: str,
    output_column: str,
    frequency: str = "MS",
) -> pd.DataFrame:
    """
    Load a local CSV file into a monthly time-series frame.

    This is the preferred path when DOEB/EPPO files are downloaded manually from
    the official portals and stored in the project workspace for offline use.
    """

    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df[date_column], errors="coerce")
    df[output_column] = pd.to_numeric(df[value_column], errors="coerce")

    return (
        df[["Date", output_column]]
        .dropna()
        .sort_values("Date")
        .drop_duplicates(subset=["Date"])
        .set_index("Date")
        .asfreq(frequency)
    )


def load_owid_energy_dataset(
    *,
    csv_path_or_url: str = "https://owid-public.owid.io/data/energy/owid-energy-data.csv",
    country: str = "Thailand",
) -> pd.DataFrame:
    """
    Load OWID energy data for optional exploratory features.

    OWID is annual rather than monthly, so this frame is not directly suitable as
    a target in the VAR pipeline. It is kept here for allowed-dataset compliance
    and future feature engineering.
    """

    df = pd.read_csv(csv_path_or_url)
    country_df = df[df["country"] == country].copy()
    if country_df.empty:
        raise ValueError(f"No OWID energy data found for country={country}.")

    country_df["Date"] = pd.to_datetime(country_df["year"].astype(str) + "-01-01")
    keep_columns = [column for column in ("oil_consumption", "gas_consumption", "electricity_demand") if column in country_df.columns]
    result = country_df[["Date", *keep_columns]].set_index("Date").sort_index()
    return result


def build_real_local_targets(
    *,
    doeb_import_path: str | Path | None = None,
    doeb_diesel_path: str | Path | None = None,
    eppo_monthly_path: str | Path | None = None,
) -> tuple[pd.DataFrame | None, dict[str, str]]:
    """
    Build monthly Thailand target variables from locally downloaded official files.

    Expected local files:
    - DOEB import volume CSV with columns: Date, Value
    - DOEB diesel sales CSV with columns: Date, Value
    - Optional EPPO monthly CSV for future enrichment
    """

    source_map: dict[str, str] = {}
    frames: list[pd.DataFrame] = []

    if doeb_import_path and Path(doeb_import_path).exists():
        frames.append(
            load_timeseries_csv(
                doeb_import_path,
                date_column="Date",
                value_column="Value",
                output_column="doeb_import_volume",
            )
        )
        source_map["doeb_import_volume"] = str(Path(doeb_import_path))

    if doeb_diesel_path and Path(doeb_diesel_path).exists():
        frames.append(
            load_timeseries_csv(
                doeb_diesel_path,
                date_column="Date",
                value_column="Value",
                output_column="doeb_diesel_sales",
            )
        )
        source_map["doeb_diesel_sales"] = str(Path(doeb_diesel_path))

    # EPPO is optional in this phase because the current online simulator serves
    # the two DOEB targets above. The file path is recorded for transparency.
    if eppo_monthly_path and Path(eppo_monthly_path).exists():
        source_map["eppo_monthly"] = str(Path(eppo_monthly_path))

    if len(frames) < 2:
        return None, source_map

    local_df = frames[0]
    for frame in frames[1:]:
        local_df = local_df.join(frame, how="inner")

    local_df.index.name = "Date"
    return local_df.sort_index().dropna(), source_map


def build_synthetic_local_targets(
    global_df: pd.DataFrame,
    *,
    lag_months: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Fallback generator when real DOEB/EPPO files are not yet present locally.
    """

    rng = np.random.default_rng(seed)
    index = global_df.index
    n_periods = len(index)
    month_idx = np.arange(n_periods)

    brent_zscore = (
        (global_df["eia_brent_price"] - global_df["eia_brent_price"].mean())
        / global_df["eia_brent_price"].std(ddof=0)
    )
    natgas_zscore = (
        (global_df["eia_natural_gas_price"] - global_df["eia_natural_gas_price"].mean())
        / global_df["eia_natural_gas_price"].std(ddof=0)
    )

    brent_lag_signal = brent_zscore.shift(lag_months).fillna(0.0)
    natgas_lag_signal = natgas_zscore.shift(lag_months).fillna(0.0)

    import_trend = np.linspace(86, 92, n_periods)
    import_seasonality = 4.2 * np.sin(2 * np.pi * month_idx / 12)
    import_noise = rng.normal(loc=0.0, scale=1.7, size=n_periods)

    diesel_trend = np.linspace(58, 66, n_periods)
    diesel_seasonality = 3.4 * np.cos(2 * np.pi * month_idx / 12)
    diesel_noise = rng.normal(loc=0.0, scale=1.5, size=n_periods)

    doeb_import_volume = (
        import_trend
        + import_seasonality
        - 5.8 * brent_lag_signal.to_numpy()
        - 2.4 * natgas_lag_signal.to_numpy()
        + import_noise
    )
    doeb_diesel_sales = (
        diesel_trend
        + diesel_seasonality
        - 3.1 * brent_lag_signal.to_numpy()
        - 1.3 * natgas_lag_signal.to_numpy()
        + 0.24 * doeb_import_volume
        + diesel_noise
    )

    target_df = pd.DataFrame(
        {
            "doeb_import_volume": np.clip(doeb_import_volume, a_min=1.0, a_max=None),
            "doeb_diesel_sales": np.clip(doeb_diesel_sales, a_min=1.0, a_max=None),
        },
        index=index,
    )
    target_df.index.name = "Date"
    return target_df
