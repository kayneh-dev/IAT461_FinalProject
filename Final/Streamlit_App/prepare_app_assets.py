from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


DATA_FILE = Path(__file__).with_name("sturgeon-pit-tag-dat.csv")

raw_df = pd.read_csv(DATA_FILE, low_memory=False)
raw_df = raw_df.rename(columns={
    "FishID": "fish_id",
    "Tag number": "tag_number",
    "Tag type": "tag_type",
    "Year class": "year_class",
    "Confirmed hatchery-release?": "hatchery_status",
    "Capture date": "capture_date",
    "Fork length": "fork_length",
    "Girth": "girth",
    "Waterbody": "waterbody",
    "Release date": "release_date",
    "RKM_Round": "rkm_round"
})

df = raw_df.copy()
df["capture_date"] = pd.to_datetime(df["capture_date"], errors="coerce")
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
df["fork_length_clean"] = df["fork_length"].where(df["fork_length"] <= 400)

capture_rows = df[df["capture_date"].notna()].copy()
capture_rows["fork_round"] = capture_rows["fork_length_clean"].round(1)
capture_rows["girth_round"] = capture_rows["girth"].round(1)
capture_rows["hatchery_confirmed"] = capture_rows["hatchery_status"].eq("Hatchery")

encounter_keys = ["fish_id", "capture_date", "waterbody", "rkm_round"]
encounters = (
    capture_rows.groupby(encounter_keys, dropna=False, as_index=False)
    .agg(
        fork_length=("fork_length_clean", "median"),
        fork_value_count=("fork_round", "nunique"),
        girth=("girth", "median"),
        hatchery_confirmed=("hatchery_confirmed", "max")
    )
)

encounters["fork_conflict"] = encounters["fork_value_count"] > 1
encounters.loc[encounters["fork_conflict"], "fork_length"] = np.nan
encounters["capture_year"] = encounters["capture_date"].dt.year
encounters["capture_month"] = encounters["capture_date"].dt.month

rkm_parts = encounters["rkm_round"].str.extract(r"(?P<rkm_start>\d+)-(?P<rkm_end>\d+)")
encounters["rkm_start"] = pd.to_numeric(rkm_parts["rkm_start"], errors="coerce")
encounters["rkm_end"] = pd.to_numeric(rkm_parts["rkm_end"], errors="coerce")
encounters["rkm_midpoint"] = (encounters["rkm_start"] + encounters["rkm_end"]) / 2

encounters["large_mature"] = (encounters["fork_length"] >= 150).astype("Int64")
encounters.loc[encounters["fork_length"].isna(), "large_mature"] = pd.NA

segment_summary = (
    encounters.groupby(["waterbody", "rkm_round"], dropna=False)
    .agg(
        encounters=("fish_id", "size"),
        unique_fish=("fish_id", "nunique"),
        fork_complete=("large_mature", "count"),
        large_count=("large_mature", "sum"),
        median_fork_length=("fork_length", "median")
    )
    .reset_index()
)
segment_summary["fork_complete_percent"] = (
    segment_summary["fork_complete"] / segment_summary["encounters"] * 100
)
segment_summary["large_share_percent"] = (
    segment_summary["large_count"] / segment_summary["fork_complete"] * 100
)
segment_summary["location"] = (
    segment_summary["waterbody"].astype(str) + " | RKM " + segment_summary["rkm_round"].astype(str)
)
segment_summary.to_csv(Path(__file__).with_name("sturgeon_segment_summary.csv"), index=False)

model_data = encounters.dropna(
    subset=["fork_length", "waterbody", "rkm_round", "rkm_midpoint", "capture_month", "capture_year"]
).copy()
model_data["large_mature"] = (model_data["fork_length"] >= 150).astype(int)
model_data["location_segment"] = (
    model_data["waterbody"].astype(str) + " | " + model_data["rkm_round"].astype(str)
)
model_data["month_sin"] = np.sin(2 * np.pi * model_data["capture_month"] / 12)
model_data["month_cos"] = np.cos(2 * np.pi * model_data["capture_month"] / 12)

splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_index, test_index = next(
    splitter.split(model_data, model_data["large_mature"], groups=model_data["fish_id"])
)
train_data = model_data.iloc[train_index].copy()
test_data = model_data.iloc[test_index].copy()

location_counts = train_data["location_segment"].value_counts()
common_locations = location_counts[location_counts >= 100].index.tolist()
reference_location = "Fraser River | 90-100"
location_categories = [reference_location] + sorted([
    location for location in common_locations if location != reference_location
]) + ["Other / sparse segment"]

for data in [train_data, test_data]:
    data["location_model"] = data["location_segment"].where(
        data["location_segment"].isin(common_locations),
        "Other / sparse segment"
    )
    data["location_model"] = pd.Categorical(data["location_model"], categories=location_categories)

train_location = pd.get_dummies(
    train_data["location_model"], prefix="location", drop_first=True, dtype=int
)
test_location = pd.get_dummies(
    test_data["location_model"], prefix="location", drop_first=True, dtype=int
).reindex(columns=train_location.columns, fill_value=0)

numeric_features = ["capture_year", "month_sin", "month_cos"]
scaler = StandardScaler()
train_numeric = scaler.fit_transform(train_data[numeric_features])
test_numeric = scaler.transform(test_data[numeric_features])

X_train = hstack([csr_matrix(train_numeric), csr_matrix(train_location.to_numpy())]).tocsr()
X_test = hstack([csr_matrix(test_numeric), csr_matrix(test_location.to_numpy())]).tocsr()
y_train = train_data["large_mature"].to_numpy()

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

bundle = {
    "model": model,
    "scaler": scaler,
    "common_locations": common_locations,
    "location_categories": location_categories,
    "location_columns": train_location.columns.tolist(),
    "numeric_features": numeric_features,
    "reference_location": reference_location,
    "minimum_location_count": 100,
    "year_min": int(model_data["capture_year"].min()),
    "year_max": int(model_data["capture_year"].max())
}
joblib.dump(bundle, Path(__file__).with_name("sturgeon_logistic_model.joblib"))

logistic_probability = model.predict_proba(X_test)[:, 1]
test_results = test_data[["waterbody", "rkm_round", "capture_month", "large_mature"]].copy()
test_results["logistic_probability"] = logistic_probability

held_out_segments = (
    test_results.groupby(["waterbody", "rkm_round"])
    .agg(
        held_out_encounters=("large_mature", "size"),
        held_out_observed_large_percent=("large_mature", lambda values: values.mean() * 100),
        logistic_mean_probability_percent=("logistic_probability", lambda values: values.mean() * 100)
    )
    .reset_index()
)

reliable_segments = segment_summary.query(
    "fork_complete >= 100 and fork_complete_percent >= 80"
).copy()
priority = reliable_segments.merge(held_out_segments, on=["waterbody", "rkm_round"], how="inner")
priority = priority[priority["held_out_encounters"] >= 100].copy()
priority["location"] = priority["waterbody"] + " | RKM " + priority["rkm_round"]
priority = priority.sort_values(
    ["logistic_mean_probability_percent", "held_out_encounters"], ascending=[False, False]
)
priority.to_csv(Path(__file__).with_name("sturgeon_priority_locations.csv"), index=False)

month_summary = (
    test_results.groupby("capture_month")
    .agg(
        held_out_encounters=("large_mature", "size"),
        observed_large_percent=("large_mature", lambda values: values.mean() * 100),
        logistic_mean_probability_percent=("logistic_probability", lambda values: values.mean() * 100)
    )
    .reset_index()
)
month_summary.to_csv(Path(__file__).with_name("sturgeon_month_summary.csv"), index=False)

print("App assets created.")
print(f"Segments: {len(segment_summary):,}")
print(f"Priority locations: {len(priority):,}")
print(f"Model locations: {len(common_locations):,}")
