"""
Build the modelling dataset for the hmmTMB R script.
bin and screen the data based on configuration
STEP THREE (last step)

"""

import os

import numpy as np
import pandas as pd

from .utils import (
    ensure_output_dir,
    read_detections,
    solar_elevation_by_bin,
    to_utc_datetime,
)

DETECTION_COLUMNS = [
    "motusTagID",
    "tsCorrected",
    "sig",
    "recvDeployName",
    "age",
    "sex",
]

OUTPUT_COLUMNS = [
    "tag_id",
    "age",
    "sex",
    "tsCorrected",
    "local_time",
    "date",
    "period",
    "receiver",
    "sig_mean_db",
    "sig_var_db",
    "sig_sd_db",
    "n_detections",
    "doy",
    "track_id",
]


def create_hmm_dataset(
    parquet,
    receiver_summary,
    output_dir,
    latitude,
    longitude,
    timezone,
    species_id,
    tag_ids,
    bin_size,
    solar_threshold,
    periods,
    min_readings_per_bin,
    exclude_insufficient,
    require_age_sex,
    sd_cut,
    max_frac_low,
    min_bins,
    min_days,
    output_name,
    screening_name,
    verbose=True,
):

    ensure_output_dir(output_dir)

    if min_readings_per_bin < 2:
        raise ValueError(
            "`min_readings_per_bin` must be at least 2 to calculate variance. "

        )

    if max_frac_low >= 1.0:
        raise ValueError(
            "`max_frac_low` must be below 1.0"
        )

    summary = _load_receiver_summary(receiver_summary)
    lookup = _build_top_receiver_lookup(
        summary,
        exclude_insufficient=exclude_insufficient,
        verbose=verbose,
    )

    if tag_ids is None:
        tag_ids = sorted(summary["tag_id"].unique())
    tag_ids = [int(t) for t in tag_ids]

    if verbose:
        print(f"\nBuilding HMM dataset for {len(tag_ids)} tag(s)")

    df = _load_detections(
        parquet=parquet,
        latitude=latitude,
        longitude=longitude,
        timezone=timezone,
        species_id=species_id,
        tag_ids=tag_ids,
        bin_size=bin_size,
        solar_threshold=solar_threshold,
        verbose=verbose,
    )

    # we can remove unwanted period's detections here
    if periods is not None:
        periods = [str(p).lower() for p in periods]
        before = len(df)
        df = df[df["period"].isin(periods)]

        if verbose:
            print(
                f"Kept {len(df):,} of {before:,} detections "
                f"in period(s): {', '.join(periods)}"
            )

        if df.empty:
            raise ValueError(f"No detections in period(s) {periods}.")

    # One (age, sex) per tag rather than per bin
    tag_meta, meta_missing = _tag_age_sex(df, verbose=verbose)

    # bin data
    frames = []
    skipped = {}

    for tag_id in tag_ids:
        tag_df = df[df["tag_id"] == tag_id]

        if tag_df.empty:
            skipped[tag_id] = "no detections in raw file"
            if verbose:
                print(f"  tag {tag_id}: skipped ({skipped[tag_id]})")
            continue

        filtered = _filter_to_top_receiver(tag_df, lookup)

        if filtered.empty:
            skipped[tag_id] = "no detections matched a top receiver"
            if verbose:
                print(f"  tag {tag_id}: skipped ({skipped[tag_id]})")
            continue

        binned = _aggregate_bins(filtered, min_readings_per_bin)

        if binned.empty:
            skipped[tag_id] = (
                f"no bins with >= {min_readings_per_bin} detections"
            )
            if verbose:
                print(f"  tag {tag_id}: skipped ({skipped[tag_id]})")
            continue

        frames.append(binned)

        if verbose:
            print(
                f"  tag {tag_id}: {len(binned):,} bins across "
                f"{binned['date'].nunique():,} days"
            )

    if not frames:
        raise ValueError(
            "No tag produced usable bins. Check that `solar_threshold` "
            "matches the value used in `top_receivers()`."
        )

    dataset = pd.concat(frames, ignore_index=True)

    before = len(dataset)
    dataset = dataset.merge(
        tag_meta, on="tag_id", how="left", validate="many_to_one"
    )
    if len(dataset) != before:
        raise RuntimeError(
            f"age/sex merge changed row count: {before:,} -> {len(dataset):,}"
        )

    # screen sparse and suspected dropped tags
    tags, dropped = _screen_tags(
        dataset,
        meta_missing=meta_missing,
        require_age_sex=require_age_sex,
        sd_cut=sd_cut,
        max_frac_low=max_frac_low,
        min_bins=min_bins,
        min_days=min_days,
    )

    n_before = len(dataset)
    dataset = dataset[~dataset["tag_id"].isin(dropped)].copy()

    if dataset.empty:
        raise ValueError(
            "Every tag was screened out. Check `min_bins`, `min_days` and "
            "`max_frac_low` against the screening table."
        )

    dataset = dataset.sort_values(["tag_id", "tsCorrected"]).reset_index(drop=True)
    dataset = dataset[OUTPUT_COLUMNS]

    #write the final dataset file as parquet
    path = os.path.join(output_dir, output_name)
    dataset.to_parquet(path, index=False, engine="pyarrow")

    screening_path = os.path.join(output_dir, screening_name)
    tags.sort_values(["reason", "frac_low"], ascending=[True, False]).to_csv(
        screening_path, index=False
    )

    if verbose:
        _print_report(
            dataset=dataset,
            tags=tags,
            skipped=skipped,
            n_bins_before=n_before,
            sd_cut=sd_cut,
            max_frac_low=max_frac_low,
            min_bins=min_bins,
            min_days=min_days,
            path=path,
            screening_path=screening_path,
        )

    reason = tags.set_index("tag_id")["reason"]

    return {
        "dataset": dataset,
        "path": path,
        "tags": tags,
        "dropped": [int(t) for t in dropped],
        "sparse": [int(t) for t in reason[reason == "sparse"].index],
        "flat": [int(t) for t in reason[reason == "stationary"].index],
        "no_metadata": [int(t) for t in reason[reason == "no age/sex"].index],
        "tag_ids": [int(t) for t in reason[reason == ""].index],
        "skipped": skipped,
        "screening_path": screening_path,
    }


#receiver lookup from computed top receiver

#can read either part of excel book
def _load_receiver_summary(receiver_summary):
    if isinstance(receiver_summary, pd.DataFrame):
        summary = receiver_summary.copy()
    else:
        summary = pd.read_excel(receiver_summary, sheet_name="receiver_summary")

    required = {"tag_id", "receiver", "date", "period", "total_readings",
                "proximity_flag"}
    missing = required - set(summary.columns)

    if missing:
        raise ValueError(
            f"`receiver_summary` is missing column(s): {sorted(missing)}"
        )

    if "insufficient_data" not in summary.columns:
        summary["insufficient_data"] = False

    summary["date"] = pd.to_datetime(summary["date"]).dt.date
    summary["proximity_flag"] = summary["proximity_flag"].astype(bool)
    summary["insufficient_data"] = summary["insufficient_data"].astype(bool)

    return summary

#choose the winning top receiver
#IN CASE OF TIE: tie is broken by receiver name so that repeated runs on the same data select the same receiver
def _build_top_receiver_lookup(summary, exclude_insufficient, verbose):
    eligible = summary[~summary["proximity_flag"]]

    if exclude_insufficient:
        before = len(eligible)
        eligible = eligible[~eligible["insufficient_data"]]
        if verbose:
            dropped = before - len(eligible)
            print(
                f"Excluded {dropped:,} untested receiver-day(s) "
                f"(insufficient_data)"
            )

    if eligible.empty:
        raise ValueError(
            "Every receiver-day was excluded. Relax `r_flag_threshold` "
            "or set `exclude_insufficient=False`."
        )

    winners = (
        eligible.sort_values(
            ["total_readings", "receiver"], ascending=[False, True]
        )
        .groupby(["tag_id", "date", "period"], as_index=False)
        .first()
    )

    if verbose:
        print(
            f"Selected top receivers for {len(winners):,} "
            f"tag/day/period combination(s)"
        )

    return {
        (int(row.tag_id), row.date, str(row.period).lower()): str(row.receiver)
        for row in winners.itertuples(index=False)
    }



# arrange age and sex

#here we reduce the per-detection age and sex data to just one row per tag
#if a tag has multiple age or sex values, it will error
#age cannot change over time for same tag
def _tag_age_sex(df, verbose):
    meta = (
        df[["tag_id", "age", "sex"]]
        .replace({"": np.nan})
        .drop_duplicates()
    )

    conflicts = []
    for column in ("age", "sex"):
        counts = (
            meta.dropna(subset=[column])
            .groupby("tag_id")[column]
            .nunique()
        )
        bad = counts[counts > 1]
        for tag_id in bad.index:
            values = sorted(
                meta.loc[meta["tag_id"] == tag_id, column].dropna().unique()
            )
            conflicts.append(f"  tag {tag_id}: {column} = {values}")

    if conflicts:
        raise ValueError(
            "Conflicting age/sex within a tag:\n" + "\n".join(conflicts)
        )

    tag_meta = (
        meta.sort_values(["tag_id", "age", "sex"])
        .groupby("tag_id", as_index=False)
        .first()
    )

    missing = set(
        int(t)
        for t in tag_meta.loc[
            tag_meta["age"].isna() | tag_meta["sex"].isna(), "tag_id"
        ]
    )

    if verbose:
        print(
            f"Resolved age/sex for {len(tag_meta) - len(missing):,} of "
            f"{len(tag_meta):,} tag(s)"
        )
        if missing:
            print(
                f"  WARNING: {len(missing)} tag(s) missing age or sex: "
                f"{sorted(missing)}"
            )

    return tag_meta, missing


# match detections to tags, bins, periods


def _load_detections(
    parquet,
    latitude,
    longitude,
    timezone,
    species_id,
    tag_ids,
    bin_size,
    solar_threshold,
    verbose,
):
    columns = list(DETECTION_COLUMNS)
    if species_id is not None:
        columns.append("speciesID")

    if verbose:
        print(f"\nLoading parquet: {parquet}")

    df = read_detections(parquet, columns)

    absent = [c for c in ("age", "sex") if c not in df.columns]
    if absent:
        raise ValueError(
            f"`{parquet}` is missing column(s): {absent}. The HMM R script uses "
            "age and sex."
        )

    if species_id is not None:
        df = df[df["speciesID"] == species_id]
        df = df.drop(columns=["speciesID"])

    df = df[df["motusTagID"].isin(tag_ids)]
    df = df.rename(columns={"motusTagID": "tag_id", "recvDeployName": "receiver"})

    df["tag_id"] = df["tag_id"].astype("int32")
    df["sig"] = df["sig"].astype("float32")
    df["receiver"] = df["receiver"].astype(str)

    #age and sex will be objects for merging purpooses
    for column in ("age", "sex"):
        df[column] = df[column].astype("object").str.strip()

    df["datetime"] = to_utc_datetime(df["tsCorrected"])
    df = df.dropna(subset=["datetime", "sig"])

    if df.empty:
        raise ValueError("No detections found for the requested tags.")

    #drop duplicates
    #very important as there were many duplicates found in our motus data
    before = len(df)
    df = df.drop_duplicates(subset=["tag_id", "tsCorrected", "receiver", "sig"])

    if verbose and before > len(df):
        print(f"Dropped {before - len(df):,} duplicate detection(s)")

    df["time_bin"] = df["datetime"].dt.floor(bin_size)

    if verbose:
        print("Computing solar elevation...")

    df = solar_elevation_by_bin(df, "time_bin", latitude, longitude)

    df["period"] = np.where(
        df["solar_elevation"] > solar_threshold, "day", "night"
    )

    df["local_time"] = df["datetime"].dt.tz_convert(timezone)
    df["date"] = df["local_time"].dt.date

    return df

#keep only top receivers detections
def _filter_to_top_receiver(tag_df, lookup):
    keys = list(
        zip(
            tag_df["tag_id"].astype(int),
            tag_df["date"],
            tag_df["period"],
        )
    )

    expected = pd.Series(
        [lookup.get(key) for key in keys], index=tag_df.index, dtype="object"
    )

    return tag_df[expected.notna() & (tag_df["receiver"] == expected)].copy()

#one row per bin with its mean signal and standard deviation
def _aggregate_bins(df, min_readings_per_bin):
 #Signal is stored as float32 to save memory, but variance is  float64  to keep rounding error out of sig_sd_db
    df = df.copy()
    df["sig"] = df["sig"].astype("float64")

    binned = (
        df.groupby(["tag_id", "time_bin"], observed=True)
        .agg(
            tsCorrected=("tsCorrected", "first"),
            local_time=("local_time", "first"),
            date=("date", "first"),
            period=("period", "first"),
            receiver=("receiver", "first"),
            sig_mean_db=("sig", "mean"),
            sig_var_db=("sig", "var"),
            n_detections=("sig", "size"),
        )
        .reset_index()
    )

    binned = binned[binned["n_detections"] >= min_readings_per_bin]
    binned = binned.dropna(subset=["sig_var_db"])

    if binned.empty:
        return binned

   #calculate standard dev
    binned["sig_sd_db"] = np.sqrt(binned["sig_var_db"])

    # Day of year mapped to date
    binned["doy"] = pd.to_datetime(binned["date"]).dt.dayofyear

    # One track per tag/day/period so the Markov chain resets each morning -- we are only using daytime values now
    binned["track_id"] = (
        binned["tag_id"].astype(str)
        + "_"
        + binned["date"].astype(str)
        + "_"
        + binned["period"].astype(str)
    )

    binned["local_time"] = binned["local_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    binned["date"] = binned["date"].astype(str)

    return binned.sort_values("tsCorrected").reset_index(drop=True)


# Screening tags


def _screen_tags(dataset, meta_missing, require_age_sex,
                 sd_cut, max_frac_low, min_bins, min_days):
    day = dataset[dataset["period"] == "day"]

    if day.empty:
        raise ValueError(
            "The dataset contains no day bins, so no tag can be assessed."
        )

    tags = (
        day.groupby("tag_id")
        .agg(
            n_bins=("sig_sd_db", "size"),
            n_days=("date", "nunique"),
            median_sd=("sig_sd_db", "median"),
            frac_low=("sig_sd_db", lambda s: (s < sd_cut).mean()),
        )
        .reset_index()
    )

    tags["age"] = tags["tag_id"].map(
        dataset.drop_duplicates("tag_id").set_index("tag_id")["age"]
    )
    tags["sex"] = tags["tag_id"].map(
        dataset.drop_duplicates("tag_id").set_index("tag_id")["sex"]
    )

    sparse = (tags["n_bins"] < min_bins) | (tags["n_days"] < min_days)

    #screen frac low AFTER sparse to avoid weird frac low designations on tags that barley have detections in the first place
    flat = ~sparse & (tags["frac_low"] > max_frac_low)

    tags["reason"] = ""
    tags.loc[sparse, "reason"] = "sparse"
    tags.loc[flat, "reason"] = "stationary"

    if require_age_sex:
        no_meta = (
            (tags["reason"] == "")
            & tags["tag_id"].astype(int).isin(meta_missing)
        )
        tags.loc[no_meta, "reason"] = "no age/sex"

    tags["drop"] = tags["reason"] != ""
    dropped = [int(t) for t in tags.loc[tags["drop"], "tag_id"]]

    return tags, dropped


# Print to console
#comment out if its too bulky. I like it as a quick check

HEADER = (f"  {'tag_id':>10} {'frac_low':>10} {'median_sd':>10} "
          f"{'day_bins':>10} {'days':>6}")


def _rows(subset):
    for row in subset.itertuples(index=False):
        print(
            f"  {int(row.tag_id):>10} {row.frac_low:>10.3f} "
            f"{row.median_sd:>10.3f} {int(row.n_bins):>10,} "
            f"{int(row.n_days):>6}"
        )


def _print_report(dataset, tags, skipped, n_bins_before, sd_cut,
                  max_frac_low, min_bins, min_days, path, screening_path):
    print("\n" + "=" * 80)
    print(f"{'HMM DATASET':^80}")
    print("=" * 80)

    print(f"{'sd_cut:':<24} {sd_cut}")
    print(f"{'max_frac_low:':<24} {max_frac_low}")
    print(f"{'min_bins / min_days:':<24} {min_bins} / {min_days}")

    sparse = tags[tags["reason"] == "sparse"]
    flat = tags[tags["reason"] == "stationary"]
    no_meta = tags[tags["reason"] == "no age/sex"]
    kept = tags[~tags["drop"]]

    print(f"\n{'Tags binned:':<24} {len(tags):,}")
    print(f"{'Dropped sparse:':<24} {len(sparse):,}")
    print(f"{'Dropped stationary:':<24} {len(flat):,}")
    if len(no_meta):
        print(f"{'Dropped no age/sex:':<24} {len(no_meta):,}")
    print(f"{'Tags kept:':<24} {len(kept):,}")
    print(f"{'Bins:':<24} {n_bins_before:,} -> {len(dataset):,}")
    print(f"{'Tracks:':<24} {dataset['track_id'].nunique():,}")
    print(f"{'Date range:':<24} {dataset['date'].min()} -> {dataset['date'].max()}")

    for period in ("day", "night"):
        subset = dataset[dataset["period"] == period]
        if not subset.empty:
            print(f"  {period:<8} {len(subset):,} bins")

    if len(flat):
        print(f"\nStationary ({len(flat)}):")
        print(HEADER)
        _rows(flat.sort_values("n_bins", ascending=False))

    if len(sparse):
        print(f"\nSparse ({len(sparse)}), largest first:")
        print(HEADER)
        _rows(sparse.nlargest(10, "n_bins"))
        if len(sparse) > 10:
            print(f"  ... and {len(sparse) - 10} more")

    # print  kept tags that came closest to being dropped by each rule to see how your threshold is performing
    if not kept.empty:
        print("\nClosest kept, by frac_low:")
        print(HEADER)
        _rows(kept.nlargest(5, "frac_low"))

        print("\nClosest kept, by day bins:")
        print(HEADER)
        _rows(kept.nsmallest(5, "n_bins"))

    #counts individuals  per age × sex cell
    per_tag = dataset.drop_duplicates("tag_id")[["tag_id", "age", "sex"]]
    composition = (
        per_tag.groupby(["age", "sex"], dropna=False)
        .size()
        .reset_index(name="tags")
        .sort_values(["age", "sex"])
    )

    print("\nTags per age x sex cell (kept tags only):")
    for row in composition.itertuples(index=False):
        age = "(missing)" if pd.isna(row.age) else row.age
        sex = "(missing)" if pd.isna(row.sex) else row.sex
        print(f"  {age:<10} {sex:<10} {row.tags:>4}")

    incomplete = per_tag[per_tag["age"].isna() | per_tag["sex"].isna()]
    if not incomplete.empty:
        print(
            f"\n  WARNING: {len(incomplete)} kept tag(s) have no age or sex: "
            f"{sorted(int(t) for t in incomplete['tag_id'])}"
        )
        print("  The R stage drops these rows, so the fitted sample is smaller")
        print("  than the counts above. Set require_age_sex=True to drop them here.")

    if skipped:
        print(f"\nProduced no bins ({len(skipped)}):")
        for tag_id, reason in skipped.items():
            print(f"  - {tag_id}: {reason}")

    print("=" * 80)
    print(f"Dataset:   {path}")
    print(f"Screening: {screening_path}")