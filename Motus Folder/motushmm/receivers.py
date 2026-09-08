"""
Select the top receiver for each day and night

flag receivers that have correlated signal and SD when there is sufficient data
STEP 2
"""

import os

import numpy as np
import pandas as pd

from .utils import (
    drop_receivers,
    ensure_output_dir,
    read_detections,
    solar_elevation_by_bin,
    to_utc_datetime,
)

#minimal columns needed for calculations from raw parquet
RECEIVER_COLUMNS = [
    "motusTagID",
    "recvDeployName",
    "tsCorrected",
    "sig",
]
#columns in output file
SUMMARY_COLUMNS = [
    "tag_id",
    "receiver",
    "date",
    "period",
    "month",
    "total_readings",
    "n_bins",
    "mean_sig",
    "mean_bin_std",
    "r_pearson",
    "r_spearman",
    "slope",
    "intercept",
    "proximity_flag",
    "insufficient_data",
]

#calculate top receivers using inputs from configuration in run
def top_receivers(
    parquet,
    output_dir,
    latitude,
    longitude,
    timezone,
    species_id,
    tag_ids,
    bin_size,
    solar_threshold,
    min_readings_per_bin,
    r_flag_threshold,
    min_bins_for_correlation,
    exclude_receivers,
    start,
    end,
    output_name,
    verbose=True
):

    ensure_output_dir(output_dir)

#prepare parquet before computation and statistical testing
    df = _load_and_prepare(
        parquet, latitude, longitude, timezone, species_id,
        tag_ids, bin_size, exclude_receivers, start, end, verbose,
    )
#prepare for summary building
    parts = []

    for period in ("day", "night"):
        bins = compute_bin_stats(df, period, solar_threshold, min_readings_per_bin, bin_size, verbose)
        part = build_summary(bins, period, r_flag_threshold, min_bins_for_correlation)
        if not part.empty:
            parts.append(part)

    if not parts:
        raise ValueError(
            "No receiver-day summaries were produced. Check "
            "`min_readings_per_bin`, `solar_threshold`, and the date range."
        )

    summary = pd.concat(parts, ignore_index=True)
    summary = summary.sort_values(
        ["tag_id", "date", "period", "total_readings"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)

    path = save_excel(summary, output_dir, output_name)

    if verbose:
        _print_receiver_report(summary, path)

    return {"summary": summary, "path": path}



# load and prepare parquet for computation and statistical testing


def _load_and_prepare(
    parquet,
    latitude,
    longitude,
    timezone,
    species_id,
    tag_ids,
    bin_size,
    exclude_receivers,
    start,
    end,
    verbose,
):
#Read detections, filter, bin, attach solar elevation
    columns = list(RECEIVER_COLUMNS)
    if species_id is not None:
        columns.append("speciesID")

    if verbose:
        print(f"\nLoading parquet: {parquet}")

    df = read_detections(parquet, columns)

    if verbose:
        print(f"Loaded {len(df):,} detections")

    df["motusTagID"] = df["motusTagID"].astype("int32")
    df["recvDeployName"] = df["recvDeployName"].astype("category")
    df["sig"] = df["sig"].astype("float32")

    if species_id is not None:
        df = df[df["speciesID"] == species_id]
        df = df.drop(columns=["speciesID"])
        if verbose:
            print(f"Filtered to speciesID {species_id}: {len(df):,} detections")

    if tag_ids is not None:
        df = df[df["motusTagID"].isin([int(t) for t in tag_ids])]
        if verbose:
            print(f"Filtered to {len(list(tag_ids))} tag(s): {len(df):,} detections")

    if verbose:
        print("Converting timestamps...")

    df["datetime"] = to_utc_datetime(df["tsCorrected"])
    df = df.dropna(subset=["datetime", "sig"])

    if verbose:
        print(f"Found {df['motusTagID'].nunique():,} unique tags")

    df = drop_receivers(df, "recvDeployName", exclude_receivers)

    if start:
        df = df[df["datetime"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df["datetime"] <= pd.Timestamp(end, tz="UTC")]

    if df.empty:
        raise ValueError("No detections remain after filtering.")

    if verbose:
        print(f"Remaining detections: {len(df):,}")

    df["time_bin"] = df["datetime"].dt.floor(bin_size)

    # convert to local calendar day so that day and night groupings don't get messed up by timezone differences.
    df["local_time"] = df["datetime"].dt.tz_convert(timezone)
    df["date"] = df["local_time"].dt.date

    # Formatted directly so that it is a timezone aware column
    df["month"] = df["local_time"].dt.strftime("%Y-%m")

    if verbose:
        print("Computing solar elevation...")

    df = solar_elevation_by_bin(df, "time_bin", latitude, longitude)

    return df


# Bin statistics


def compute_bin_stats(
    df,
    period,
    solar_threshold,
    min_readings_per_bin,
    bin_size,
    verbose=True,
):

#aggregate detections into tag/receiver bin and separate into day and night period

    if verbose:
        print(f"\nAggregating {bin_size} bins ({period})...")

    grouped = (
        df.groupby(
            ["motusTagID", "recvDeployName", "time_bin", "date", "month"],
            observed=True,
        )
        .agg(
            n_readings=("sig", "size"),
            mean_signal=("sig", "mean"),
            std_dev=("sig", "std"),
            solar_elevation=("solar_elevation", "mean"),
        )
        .reset_index()
    )

    grouped = grouped[grouped["n_readings"] >= min_readings_per_bin]

    # drop single reading bins as you cannot compute standard deviation from one reading
    grouped = grouped.dropna(subset=["std_dev"])

    if period == "day":
        grouped = grouped[grouped["solar_elevation"] > solar_threshold]
    else:
        grouped = grouped[grouped["solar_elevation"] <= solar_threshold]

    if verbose:
        print(f"Valid {period} bins: {len(grouped):,}")

    return grouped


# Proximity statistics


def daily_proximity_stats(
    bin_stats,
    r_flag_threshold=0.5,
    min_bins_for_correlation=5,
):
#Proximity statistics for one tag/receiver/day/period group
#correlations only computed if there is sufficient amount of data based on configuration specification

    n_bins = len(bin_stats)
    total_readings = int(bin_stats["n_readings"].sum()) if n_bins else 0

    if n_bins < min_bins_for_correlation:
        return {
            "n_bins": n_bins,
            "total_readings": total_readings,
            "mean_sig": bin_stats["mean_signal"].mean() if n_bins else np.nan,
            "mean_bin_std": bin_stats["std_dev"].mean() if n_bins else np.nan,
            "r_pearson": np.nan,
            "r_spearman": np.nan,
            "slope": np.nan,
            "intercept": np.nan,
            "proximity_flag": False,
            "insufficient_data": True,
        }

    x = bin_stats["mean_signal"]
    y = bin_stats["std_dev"]

    # A constant column makes the correlation undefined (zero variance in the denominator).
# This is NOT the same as correlation = zero, so it is reported as
    # untested
    x_varies = x.nunique() > 1
    y_varies = y.nunique() > 1

    if not (x_varies and y_varies):
        return {
            "n_bins": n_bins,
            "total_readings": total_readings,
            "mean_sig": x.mean(),
            "mean_bin_std": y.mean(),
            "r_pearson": np.nan,
            "r_spearman": np.nan,
            "slope": np.nan,
            "intercept": np.nan,
            "proximity_flag": False,
            "insufficient_data": True,
        }

    r_pearson = x.corr(y, method="pearson") #parametric test
    r_spearman = x.corr(y, method="spearman") #non parametric test

    slope, intercept = np.polyfit(x, y, 1)
#if either parametric or non parametric correlation is above threshold, it gets flagged
#could be uncessary to look at both
    flag = bool(
        (pd.notna(r_pearson) and abs(r_pearson) > r_flag_threshold)
        or (pd.notna(r_spearman) and abs(r_spearman) > r_flag_threshold)
    )

    return {
        "n_bins": n_bins,
        "total_readings": total_readings,
        "mean_sig": x.mean(),
        "mean_bin_std": y.mean(),
        "r_pearson": r_pearson,
        "r_spearman": r_spearman,
        "slope": slope,
        "intercept": intercept,
        "proximity_flag": flag,
        "insufficient_data": False,
    }


# Build summary file

def build_summary(
    bin_stats,
    period,
    r_flag_threshold=0.5,
    min_bins_for_correlation=5,
):
#one row per tag/receiver/day
    if bin_stats.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    rows = []

    for (tag_id, receiver, date, month), group in bin_stats.groupby(
        ["motusTagID", "recvDeployName", "date", "month"], observed=True
    ):
        stats = daily_proximity_stats(
            group,
            r_flag_threshold=r_flag_threshold,
            min_bins_for_correlation=min_bins_for_correlation,
        )

        rows.append(
            {
                "tag_id": int(tag_id),
                "receiver": str(receiver),
                "date": date,
                "period": period,
                "month": month,
                **stats,
            }
        )

    summary = pd.DataFrame(rows)
    return summary[SUMMARY_COLUMNS]



# save the excel
#two sheets: summary with all rows, top receivers with only selected winning receiver

def save_excel(summary, output_dir, output_name="daily_receiver_proximity.xlsx"):

    ensure_output_dir(output_dir)
    path = os.path.join(output_dir, output_name)

    eligible = summary[~summary["proximity_flag"]]

    if eligible.empty:
        top = pd.DataFrame(columns=summary.columns)
    else:
        top = (
            eligible.sort_values("total_readings", ascending=False)
            .groupby(["tag_id", "date", "period"], as_index=False)
            .first()
        )

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="receiver_summary", index=False)
        top.to_excel(writer, sheet_name="top_receivers", index=False)

    return path

#print the report in console
def _print_receiver_report(summary, path):
    n_flagged = int(summary["proximity_flag"].sum())
    n_insufficient = int(summary["insufficient_data"].sum())
    n_total = len(summary)

    print("\n" + "=" * 80)
    print(f"{'RECEIVER PROXIMITY SUMMARY':^80}")
    print("=" * 80)
    print(f"{'Receiver-days:':<30} {n_total:,}")
    print(f"{'Tags:':<30} {summary['tag_id'].nunique():,}")
    print(f"{'Receivers:':<30} {summary['receiver'].nunique():,}")
    print(
        f"{'Proximity flagged:':<30} {n_flagged:,} "
        f"({_pct(n_flagged, n_total)})"
    )
    print(
        f"{'Insufficient data (untested):':<30} {n_insufficient:,} "
        f"({_pct(n_insufficient, n_total)})"
    )

    for period in ("day", "night"):
        subset = summary[summary["period"] == period]
        if not subset.empty:
            print(
                f"  {period:<10} {len(subset):,} receiver-days, "
                f"{int(subset['proximity_flag'].sum()):,} flagged"
            )

    print("=" * 80)
    print(f"Saved to: {path}")


def _pct(n, total):
    return f"{100 * n / total:.1f}%" if total else "n/a"