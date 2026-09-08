"""
audit the motus dataset

writes one row per tag, with columns motusTagID,	speciesID,	age,	sex,	n_detections,	first_detection,	last_detection,	n_days,	n_receivers,	receivers
This file is for your knowledge, the audit isn't used by the other two scripts.
"""

import os

import pandas as pd

from .utils import ensure_output_dir, read_detections, to_utc_datetime

AUDIT_COLUMNS = [
    "motusTagID",
    "tsCorrected",
    "recvDeployName",
    "speciesID",
    "age",
    "sex",
]


def audit_dataset(
    parquet,
    output_dir,
    timezone,
    species_id=None,
    output_name="audit_summary.csv",
    verbose=True, #gives you updates as the program runs
):

    ensure_output_dir(output_dir)

    if verbose:
        print(f"\nLoading parquet: {parquet}")

    df = read_detections(parquet, AUDIT_COLUMNS)

    if verbose:
        print(f"Loaded {len(df):,} detections")

    df["motusTagID"] = df["motusTagID"].astype("int64")
    df["speciesID"] = df["speciesID"].astype("Int64")

    #receiver names saved as strings so they can be combined into lists
    df["recvDeployName"] = df["recvDeployName"].astype(str)

    #convert blanks to NA (missing)
    for col in ("age", "sex"):
        df[col] = df[col].astype("string").str.strip().replace("", pd.NA)

    if species_id is not None:
        df = df[df["speciesID"] == species_id]
        if verbose:
            print(f"Filtered to speciesID {species_id}: {len(df):,} detections")

    df["datetime"] = to_utc_datetime(df["tsCorrected"])
    df = df.dropna(subset=["motusTagID", "datetime"])

    if df.empty:
        raise ValueError(
            "No detections remain after filtering. "
            "Check `species_id` and the timestamp column."
        )

    if verbose:
        #print(f"Found {df['motusTagID'].nunique():,} unique tag IDs")
        print("Computing summaries...")

    summary = (
        df.groupby("motusTagID", observed=True)
        .agg(
            n_detections=("motusTagID", "size"),
            first_detection=("datetime", "min"),
            last_detection=("datetime", "max"),
            speciesID=("speciesID", _first_non_null),
            age=("age", _first_non_null),
            sex=("sex", _first_non_null),
            receivers=("recvDeployName", _sorted_unique),
        )
        .reset_index()
    )

    # make sure timezone is correct before reporting these values so that days do not get messed up
    first_local = summary["first_detection"].dt.tz_convert(timezone)
    last_local = summary["last_detection"].dt.tz_convert(timezone)

    summary["n_days"] = (
        last_local.dt.normalize() - first_local.dt.normalize()
    ).dt.days + 1

    summary["n_receivers"] = summary["receivers"].str.len()

    summary["first_detection"] = first_local.dt.strftime("%Y-%m-%d")
    summary["last_detection"] = last_local.dt.strftime("%Y-%m-%d")

    summary["receivers"] = summary["receivers"].apply(
        lambda names: ", ".join(map(str, names))
    )

    for col in ("speciesID", "age", "sex"):
        summary[col] = _fill_unknown(summary[col])

    summary = summary[
        [
            "motusTagID",
            "speciesID",
            "age",
            "sex",
            "n_detections",
            "first_detection",
            "last_detection",
            "n_days",
            "n_receivers",
            "receivers",
        ]
    ].sort_values("motusTagID")

    path = os.path.join(output_dir, output_name)
    summary.to_csv(path, index=False)

    if verbose:
        print_audit(summary)
        print(f"\nSummary saved to: {path}")

    return {
        "summary": summary,
        "path": path,
        "tag_ids": [int(t) for t in summary["motusTagID"]],
    }

#look for first non blank value
def _first_non_null(values):
    clean = values.dropna()
    return clean.iloc[0] if len(clean) else pd.NA

#sort non blanks in age and sex
def _sorted_unique(values):
    return sorted(set(values.dropna()))

#replace blanks with unknown, good for your reference
def _fill_unknown(series):
    return series.astype("object").where(series.notna(), "Unknown")

#prints quick sanity check summary in console and indicates where file is saved
def print_audit(summary):
    print("=" * 80)
    print(f"{'RAW DATA AUDIT':^80}")
    print("=" * 80)

    print(f"{'Tag IDs found:':<20} {len(summary):,}")
    print(f"{'Total detections:':<20} {summary['n_detections'].sum():,}")
    print(f"{'Dataset start:':<20} {summary['first_detection'].min()}")
    print(f"{'Dataset end:':<20} {summary['last_detection'].max()}")
    print("=" * 80)