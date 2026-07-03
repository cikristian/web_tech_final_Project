"""
predictor/services.py
----------------------
Loads the trained models once (module-level, not per-request) and generates
a next-day prediction from the latest WeatherReading row fetched by data_api.
"""

import os
import joblib
import numpy as np
from django.conf import settings

MODEL_DIR = os.path.join(settings.BASE_DIR, "predictor", "model_artifacts")

# Load once at import time -- NOT inside the view/function that runs per request.
temp_model = joblib.load(os.path.join(MODEL_DIR, "temp_model.pkl"))
condition_model = joblib.load(os.path.join(MODEL_DIR, "condition_model.pkl"))
wind_model = joblib.load(os.path.join(MODEL_DIR, "wind_model.pkl"))
condition_encoder = joblib.load(os.path.join(MODEL_DIR, "condition_encoder.pkl"))
wind_encoder = joblib.load(os.path.join(MODEL_DIR, "wind_encoder.pkl"))
feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
temp_scaler = joblib.load(os.path.join(MODEL_DIR, "temp_scaler.pkl"))
temp_uses_scaler = joblib.load(os.path.join(MODEL_DIR, "temp_uses_scaler.pkl"))


def build_feature_row(latest_reading, previous_readings):
    """
    latest_reading: the most recent WeatherReading row (model instance or dict)
    previous_readings: queryset/list of prior readings (>=7, oldest->newest)
                        used to compute rolling averages, ordered by date.

    Returns a single-row feature array in the exact column order the model
    was trained on (feature_columns.pkl).
    """
    temps = [r.temperature for r in previous_readings] + [latest_reading.temperature]

    row = {
        "temperature_celsius": latest_reading.temperature,
        "humidity": latest_reading.humidity,
        "pressure_mb": latest_reading.pressure,
        "precip_mm": latest_reading.precipitation,
        "cloud": latest_reading.cloud_cover,
        "wind_kph": latest_reading.wind_speed,
        "uv_index": getattr(latest_reading, "uv_index", 0.0),
        "temp_change_1d": temps[-1] - temps[-2] if len(temps) >= 2 else 0.0,
        "temp_roll3": float(np.mean(temps[-3:])),
        "temp_roll7": float(np.mean(temps[-7:])),
        "humidity_roll3": float(np.mean(
            [r.humidity for r in previous_readings[-2:]] + [latest_reading.humidity]
        )),
        "precip_roll3": float(np.mean(
            [r.precipitation for r in previous_readings[-2:]] + [latest_reading.precipitation]
        )),
        "doy_sin": np.sin(2 * np.pi * latest_reading.timestamp.timetuple().tm_yday / 365),
        "doy_cos": np.cos(2 * np.pi * latest_reading.timestamp.timetuple().tm_yday / 365),
    }
    # Order must match feature_cols exactly
    return np.array([[row[col] for col in feature_cols]])


def predict_next_day(latest_reading, previous_readings):
    X = build_feature_row(latest_reading, previous_readings)

    if temp_uses_scaler:
        temp_pred = float(temp_model.predict(temp_scaler.transform(X))[0])
    else:
        temp_pred = float(temp_model.predict(X)[0])

    condition_pred = condition_encoder.inverse_transform(
        condition_model.predict(X)
    )[0]
    wind_pred = wind_encoder.inverse_transform(
        wind_model.predict(X)
    )[0]

    return {
        "predicted_temperature": round(temp_pred, 1),
        "predicted_condition": condition_pred,
        "predicted_wind_direction": wind_pred,
    }


# --- Example usage inside a view ---
#
# from .services import predict_next_day
# from data_api.models import WeatherReading
#
# def predictor_view(request):
#     readings = WeatherReading.objects.order_by("-timestamp")[:8]
#     latest, previous = readings[0], list(reversed(readings[1:]))
#     result = predict_next_day(latest, previous)
#
#     Prediction.objects.create(
#         input_snapshot=latest.id,
#         predicted_temperature=result["predicted_temperature"],
#         predicted_condition=result["predicted_condition"],
#         predicted_wind_direction=result["predicted_wind_direction"],
#         model_used="RandomForest+Ridge",
#     )
#     return render(request, "predictor/prediction.html", {"result": result})
