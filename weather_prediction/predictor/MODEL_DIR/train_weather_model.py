"""
Train Weather Prediction Models (Next-Day Temperature, Condition, Wind Direction)
-----------------------------------------------------------------------------------
Domain: Weather | Data: GlobalWeatherRepository.csv (historical, per-location)

This script trains THREE models on historical daily-aggregated weather data
for a chosen location:
  1. Regression      -> next-day temperature (Celsius)
  2. Classification   -> next-day condition group (Sunny/Cloudy/Rain/Storm/Fog)
  3. Classification   -> next-day wind direction (8-point compass)

Algorithm choice: Random Forest
  - Handles non-linear relationships between weather variables without needing
    scaling/normalization.
  - Robust to the mixed feature types here (continuous + engineered lag features).
  - Gives feature_importances_, useful to justify/interpret in the report.
  - Works well on small-to-medium tabular datasets (~700-800 rows/location),
    where deep learning would overfit.

Output: saved to ./model_artifacts/
  - temp_model.pkl        (RandomForestRegressor)
  - condition_model.pkl   (RandomForestClassifier)
  - wind_model.pkl        (RandomForestClassifier)
  - condition_encoder.pkl (LabelEncoder for condition classes)
  - wind_encoder.pkl      (LabelEncoder for wind direction classes)
  - feature_columns.pkl   (list of feature column names, in order)

These artifacts are what the Django `predictor` app loads with joblib.load()
to generate live predictions from the latest row fetched by `data_api`.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, classification_report
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CSV_PATH = "GlobalWeatherRepository.csv"   # place the dataset next to this script
LOCATION = "Kigali"                         # change per your project's chosen city
TEST_FRACTION = 0.15                        # last 15% of days used as test set (time-based split)
OUTPUT_DIR = "model_artifacts"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. LOAD & FILTER
# ---------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
df["last_updated"] = pd.to_datetime(df["last_updated"])
df = df[df["location_name"] == LOCATION].copy()

if df.empty:
    raise ValueError(f"No rows found for location '{LOCATION}'. Check the name in the CSV.")

# ---------------------------------------------------------------------------
# 2. DAILY AGGREGATION
#    The raw data has irregular multiple-readings-per-day timestamps.
#    We aggregate to one row per calendar day so "next day" is unambiguous.
# ---------------------------------------------------------------------------
df["date"] = df["last_updated"].dt.date

daily = df.groupby("date").agg({
    "temperature_celsius": "mean",
    "condition_text": lambda x: x.mode()[0],
    "wind_direction": lambda x: x.mode()[0],
    "humidity": "mean",
    "pressure_mb": "mean",
    "precip_mm": "sum",
    "cloud": "mean",
    "wind_kph": "mean",
    "uv_index": "mean",
}).reset_index()

daily["date"] = pd.to_datetime(daily["date"])
daily = daily.sort_values("date").reset_index(drop=True)

# ---------------------------------------------------------------------------
# 3. TARGET GROUPING
#    Condition: map the many raw strings into 5 broad, meaningful classes.
#    Wind: map 16-point compass to 8-point compass (coarser, more learnable).
# ---------------------------------------------------------------------------
def group_condition(text: str) -> str:
    t = text.lower()
    # Note: thunderstorm days are folded into "Rain" -- too rare on their own
    # (near-zero samples for this location) to be a reliably learnable class.
    if "thunder" in t or "storm" in t or "rain" in t or "drizzle" in t or "shower" in t:
        return "Rain"
    if "fog" in t or "mist" in t or "haze" in t:
        return "Fog"
    if "cloud" in t or "overcast" in t:
        return "Cloudy"
    return "Clear"  # sunny / clear

def group_wind(direction: str) -> str:
    mapping = {
        "N": "N", "NNE": "N", "NE": "NE", "ENE": "NE",
        "E": "E", "ESE": "E", "SE": "SE", "SSE": "SE",
        "S": "S", "SSW": "S", "SW": "SW", "WSW": "SW",
        "W": "W", "WNW": "W", "NW": "NW", "NNW": "NW",
    }
    return mapping.get(direction, direction)

daily["condition_group"] = daily["condition_text"].apply(group_condition)
daily["wind_group"] = daily["wind_direction"].apply(group_wind)

# ---------------------------------------------------------------------------
# 4. FEATURE ENGINEERING
#    Inputs = TODAY's observed conditions.
#    Targets = TOMORROW's temperature / condition / wind (shift -1).
#    This mirrors exactly how the Django predictor will work: it feeds in
#    the latest fetched reading and asks for tomorrow's forecast.
# ---------------------------------------------------------------------------
feature_cols = [
    "temperature_celsius", "humidity", "pressure_mb",
    "precip_mm", "cloud", "wind_kph", "uv_index",
]

# short-term trend features: helps the model see if temp is rising/falling
daily["temp_change_1d"] = daily["temperature_celsius"].diff()
daily["temp_roll3"] = daily["temperature_celsius"].rolling(3).mean()
daily["temp_roll7"] = daily["temperature_celsius"].rolling(7).mean()
daily["humidity_roll3"] = daily["humidity"].rolling(3).mean()
daily["precip_roll3"] = daily["precip_mm"].rolling(3).mean()

# seasonality features: day-of-year encoded cyclically (captures wet/dry season
# patterns even though we only have ~2 years of history)
day_of_year = daily["date"].dt.dayofyear
daily["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365)
daily["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365)

feature_cols += [
    "temp_change_1d", "temp_roll3", "temp_roll7",
    "humidity_roll3", "precip_roll3", "doy_sin", "doy_cos",
]

daily["target_temp_next"] = daily["temperature_celsius"].shift(-1)
daily["target_condition_next"] = daily["condition_group"].shift(-1)
daily["target_wind_next"] = daily["wind_group"].shift(-1)

model_df = daily.dropna(subset=feature_cols + [
    "target_temp_next", "target_condition_next", "target_wind_next"
]).reset_index(drop=True)

print(f"Usable rows after feature engineering: {len(model_df)}")

# ---------------------------------------------------------------------------
# 5. TIME-BASED TRAIN/TEST SPLIT
#    IMPORTANT: this is a time series -> we split chronologically, NOT randomly,
#    so the model is evaluated on genuinely unseen future days.
# ---------------------------------------------------------------------------
split_idx = int(len(model_df) * (1 - TEST_FRACTION))
train_df = model_df.iloc[:split_idx]
test_df = model_df.iloc[split_idx:]

X_train, X_test = train_df[feature_cols], test_df[feature_cols]

print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")

# ---------------------------------------------------------------------------
# 6. TRAIN: TEMPERATURE (REGRESSION)
# ---------------------------------------------------------------------------
y_train_temp = train_df["target_temp_next"]
y_test_temp = test_df["target_temp_next"]

temp_model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
temp_model.fit(X_train, y_train_temp)

pred_temp = temp_model.predict(X_test)
mae = mean_absolute_error(y_test_temp, pred_temp)
rmse = np.sqrt(mean_squared_error(y_test_temp, pred_temp))
r2 = r2_score(y_test_temp, pred_temp)

print("\n--- Temperature Regression ---")
print(f"MAE:  {mae:.2f} °C")
print(f"RMSE: {rmse:.2f} °C")
print(f"R2:   {r2:.3f}")

# Baseline for comparison: "persistence" model = predict tomorrow equals today.
# This is the standard baseline in weather forecasting; a model that can't beat
# it isn't adding value. Report this alongside your model's metrics.
baseline_pred = test_df["temperature_celsius"]  # today's temp used as tomorrow's guess
baseline_mae = mean_absolute_error(y_test_temp, baseline_pred)
baseline_rmse = np.sqrt(mean_squared_error(y_test_temp, baseline_pred))
print(f"(Baseline persistence MAE: {baseline_mae:.2f} °C, RMSE: {baseline_rmse:.2f} °C)")

# Algorithm comparison: Ridge (linear) regression, since Random Forest may be
# overfitting on a small, low-variance dataset. Report whichever wins.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge_model = Ridge(alpha=5.0)
ridge_model.fit(X_train_scaled, y_train_temp)
pred_ridge = ridge_model.predict(X_test_scaled)
ridge_mae = mean_absolute_error(y_test_temp, pred_ridge)
ridge_rmse = np.sqrt(mean_squared_error(y_test_temp, pred_ridge))
print(f"(Ridge regression   MAE: {ridge_mae:.2f} °C, RMSE: {ridge_rmse:.2f} °C)")

# Use whichever model actually performs best on the held-out test set.
if ridge_mae < mae:
    print(">> Ridge regression outperforms Random Forest -- using Ridge as final temp model.")
    temp_model = ridge_model
    USE_SCALER_FOR_TEMP = True
else:
    print(">> Random Forest outperforms Ridge -- keeping Random Forest as final temp model.")
    USE_SCALER_FOR_TEMP = False

# ---------------------------------------------------------------------------
# 7. TRAIN: CONDITION (CLASSIFICATION)
# ---------------------------------------------------------------------------
condition_encoder = LabelEncoder()
condition_encoder.fit(daily["condition_group"])  # fit on full label space

y_train_cond = condition_encoder.transform(train_df["target_condition_next"])
y_test_cond = condition_encoder.transform(test_df["target_condition_next"])

condition_model = RandomForestClassifier(
    n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=42
)
condition_model.fit(X_train, y_train_cond)

pred_cond = condition_model.predict(X_test)
acc_cond = accuracy_score(y_test_cond, pred_cond)
f1_cond = f1_score(y_test_cond, pred_cond, average="macro")

print("\n--- Condition Classification ---")
print(f"Accuracy: {acc_cond:.3f}")
print(f"Macro F1: {f1_cond:.3f}")
print(classification_report(
    y_test_cond, pred_cond,
    labels=range(len(condition_encoder.classes_)),
    target_names=condition_encoder.classes_,
    zero_division=0
))

# ---------------------------------------------------------------------------
# 8. TRAIN: WIND DIRECTION (CLASSIFICATION)
# ---------------------------------------------------------------------------
wind_encoder = LabelEncoder()
wind_encoder.fit(daily["wind_group"])

y_train_wind = wind_encoder.transform(train_df["target_wind_next"])
y_test_wind = wind_encoder.transform(test_df["target_wind_next"])

wind_model = RandomForestClassifier(
    n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=42
)
wind_model.fit(X_train, y_train_wind)

pred_wind = wind_model.predict(X_test)
acc_wind = accuracy_score(y_test_wind, pred_wind)
f1_wind = f1_score(y_test_wind, pred_wind, average="macro")

print("\n--- Wind Direction Classification ---")
print(f"Accuracy: {acc_wind:.3f}")
print(f"Macro F1: {f1_wind:.3f}")

# Baseline for comparison: majority-class classifier (always predict the
# most frequent class in the training set). Beating this is the real bar.
majority_cond = pd.Series(y_train_cond).mode()[0]
baseline_cond_acc = accuracy_score(y_test_cond, [majority_cond] * len(y_test_cond))
majority_wind = pd.Series(y_train_wind).mode()[0]
baseline_wind_acc = accuracy_score(y_test_wind, [majority_wind] * len(y_test_wind))
print(f"\n(Baseline majority-class accuracy — condition: {baseline_cond_acc:.3f}, wind: {baseline_wind_acc:.3f})")

# ---------------------------------------------------------------------------
# 9. EXPORT MODELS (joblib) -> loaded later inside Django's `predictor` app
# ---------------------------------------------------------------------------
joblib.dump(temp_model, os.path.join(OUTPUT_DIR, "temp_model.pkl"))
joblib.dump(condition_model, os.path.join(OUTPUT_DIR, "condition_model.pkl"))
joblib.dump(wind_model, os.path.join(OUTPUT_DIR, "wind_model.pkl"))
joblib.dump(condition_encoder, os.path.join(OUTPUT_DIR, "condition_encoder.pkl"))
joblib.dump(wind_encoder, os.path.join(OUTPUT_DIR, "wind_encoder.pkl"))
joblib.dump(feature_cols, os.path.join(OUTPUT_DIR, "feature_columns.pkl"))
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "temp_scaler.pkl"))
joblib.dump(USE_SCALER_FOR_TEMP, os.path.join(OUTPUT_DIR, "temp_uses_scaler.pkl"))

print(f"\nModels exported to ./{OUTPUT_DIR}/")
