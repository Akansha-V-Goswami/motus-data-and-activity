"""
run.py — audit raw motus data and build the filtered HMM dataset.
Set the necessary parameters in the configuration section
"""

import os
import sys
import time

import pandas as pd

from motushmm import audit_dataset, top_receivers, create_hmm_dataset

# Configuration

SOURCE_PARQUET = r"C:\Users\goswa\Downloads\amre_2022_2025_allrecv_grid_agesex_final.parquet" #raw data, should include age and sex
OUTPUT_DIR     = r"C:\Users\goswa\OneDrive\Desktop\FINAL-motus-processing-and-hmm-dataset_final"

# Location & species
LATITUDE   = 18.1096
LONGITUDE  = -77.2975
TIMEZONE   = "America/Jamaica"
SPECIES_ID = 16890          # American Redstart

#  Used for top receivers and hmm dataset building
BIN_SIZE        = "1min"
SOLAR_THRESHOLD = -12.0

# calculating top receivers:

EXCLUDE_RECEIVERS = ("Test", "Departure") #receivers we do not want to use
R_FLAG_THRESHOLD  = 0.5 #correlation threshold for pearson and spearman r
MIN_BINS_FOR_CORRELATION = 30
S2_MIN_READINGS_PER_BIN = 1 #for receiver exploration
TAG_IDS = None      # None = all tags. Or input list like [61234, 71456]
START   = None      # None = no lower date cutoff. Or input date like "YYYY-MM-DD"
END     = None      # None = no upper date cutoff. Or input date like "YYYY-MM-DD"


#VERBOSE = True
# creating hmm dataset: binning
PERIODS                 = ("day", ) #keep the comma after the period to maintain this as a tuple!!!
S3_MIN_READINGS_PER_BIN = 2 #for the final hmm dataset
EXCLUDE_INSUFFICIENT    = False #decide if you want to keep receivers who didn't have enough data to check for correlation as candidates for top daily receiver. Could be good for sparse days
REQUIRE_AGE_SEX         = False   # keep tags with missing metadata and report them. R script excludes them from the model on its own

#  creating hmm dataset: tag screening for suspected stationary or sparse tags
SD_CUT       = 1.5 #standard deviation considered suspicious
MAX_FRAC_LOW = 0.999 #percentage of days that can have low SD
MIN_BINS     = 100 #per tag
MIN_DAYS     = 5 #per tag

# Filenames
AUDIT_NAME     = "audit_summary.csv"
RECEIVER_NAME  = "daily_receiver_proximity.xlsx"
DATASET_NAME   = "hmm_dataset.parquet"
SCREENING_NAME = "tag_screening.csv"




# Helpers


def out(name):
    return os.path.join(OUTPUT_DIR, name)

#prints updates nicely in console
def banner(text):
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)



# Run the pipeline


def main():
    if not os.path.exists(SOURCE_PARQUET):
        sys.exit(f"Parquet not found: {SOURCE_PARQUET}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t_start = time.time()

    banner("STAGE 1 - audit_dataset")
    audit_dataset(
        parquet=SOURCE_PARQUET,
        output_dir=OUTPUT_DIR,
        timezone=TIMEZONE,
        species_id=SPECIES_ID,
        output_name=AUDIT_NAME,
    )

    banner("STAGE 2 - top_receivers")
    receiver_summary = top_receivers(
        parquet=SOURCE_PARQUET,
        output_dir=OUTPUT_DIR,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        timezone=TIMEZONE,
        species_id=SPECIES_ID,
        tag_ids=TAG_IDS,
        bin_size=BIN_SIZE,
        solar_threshold=SOLAR_THRESHOLD,
        min_readings_per_bin=S2_MIN_READINGS_PER_BIN,
        r_flag_threshold=R_FLAG_THRESHOLD,
        min_bins_for_correlation=MIN_BINS_FOR_CORRELATION,
        exclude_receivers=EXCLUDE_RECEIVERS,
        start=START,
        end=END,
        output_name=RECEIVER_NAME,
        #verbose=VERBOSE,
    )["summary"]

    banner("STAGE 3 - create_hmm_dataset")
    create_hmm_dataset(
        parquet=SOURCE_PARQUET,
        receiver_summary=receiver_summary,
        output_dir=OUTPUT_DIR,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        timezone=TIMEZONE,
        species_id=SPECIES_ID,
        bin_size=BIN_SIZE,
        solar_threshold=SOLAR_THRESHOLD,
        periods=PERIODS,
        min_readings_per_bin=S3_MIN_READINGS_PER_BIN,
        exclude_insufficient=EXCLUDE_INSUFFICIENT,
        require_age_sex=REQUIRE_AGE_SEX,
        sd_cut=SD_CUT,
        max_frac_low=MAX_FRAC_LOW,
        min_bins=MIN_BINS,
        min_days=MIN_DAYS,
        output_name=DATASET_NAME,
        tag_ids=TAG_IDS,
        screening_name=SCREENING_NAME,


    )
    # -- Summary -------------------------------------------------------------
    final = pd.read_parquet(out(DATASET_NAME))

    missing = [c for c in ("age", "sex") if c not in final.columns]
    if missing:
        sys.exit(
            f"{DATASET_NAME} is missing {missing}  "

        )

    banner("PIPELINE COMPLETE")
    print(f"Bins   : {len(final):,}")
    print(f"Tags   : {final['tag_id'].nunique():,}")
    print(f"Tracks : {final['track_id'].nunique():,}")
    print(f"Dates  : {final['date'].min()} -> {final['date'].max()}")

    years = pd.to_datetime(final["date"]).dt.year
    print("\nPer year:")
    for year in sorted(years.unique()):
        rows = years == year
        print(
            f"  {year}   {int(rows.sum()):>10,} bins   "
            f"{final.loc[rows, 'tag_id'].nunique():>4} tags"
        )

    print("\nTags per age x sex cell:")
    per_tag = final.drop_duplicates("tag_id")
    for (age, sex), n in per_tag.groupby(["age", "sex"], dropna=False).size().items():
        age = "(missing)" if pd.isna(age) else age
        sex = "(missing)" if pd.isna(sex) else sex
        print(f"  {age:<10} {sex:<10} {n:>4}")

    print(f"\nDataset:   {out(DATASET_NAME)}")
    print(f"Screening: {out(SCREENING_NAME)}")
    print(f"Elapsed:   {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()