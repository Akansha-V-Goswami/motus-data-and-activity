"""
Shared functions used multiple times and/or multiple scripts.
"""

import os

import pandas as pd
import pyarrow.parquet as pq
import pvlib

#create output directories even if they do not yet exist. Used 7 times across all three scripts
def ensure_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

#read only required columns from parquet files after making sure they exist
#used 6 times across all three scripts
def read_detections(parquet_path, columns):
    pf = pq.ParquetFile(parquet_path)
    available = set(pf.schema_arrow.names)

    missing = [c for c in columns if c not in available]
    if missing:
        raise ValueError(
            f"Parquet file is missing required column(s): {missing}\n"
            f"Columns present: {sorted(available)}"
        )

    return pf.read(
        columns=list(columns),
        use_pandas_metadata=True,
    ).to_pandas()

#convert unix second timestamps to timezone-aware datetimes
#used 6 times across all three scripts
def to_utc_datetime(series):
    return pd.to_datetime(series, unit="s", utc=True, errors="coerce")

#compute solar elevation for each bin once rather than for each row
#used 4 times across receiver selection and hmm dataset creation scripts
def solar_elevation_by_bin(df, bin_col, latitude, longitude):
    bins = pd.DatetimeIndex(df[bin_col].dropna().unique())

    elevation = pvlib.solarposition.get_solarposition(
        time=bins,
        latitude=latitude,
        longitude=longitude,
    )["apparent_elevation"]

    out = df.copy()
    out["solar_elevation"] = df[bin_col].map(elevation)
    return out

#drop receivers that were specified in configuration, not case sensitive!
#used twice in receivers script
def drop_receivers(df, receiver_col, patterns):
    if not patterns:
        return df

    names = df[receiver_col].astype(str)
    mask = pd.Series(False, index=df.index)

    for pattern in patterns:
        mask |= names.str.contains(pattern, case=False, na=False, regex=False)

    out = df.loc[~mask].copy()

    if isinstance(out[receiver_col].dtype, pd.CategoricalDtype):
        out[receiver_col] = out[receiver_col].cat.remove_unused_categories()

    return out