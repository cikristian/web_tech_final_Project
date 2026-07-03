def predict_next_day(latest_reading, previous_readings):
    return {
        "predicted_temperature": latest_reading.temperature + 1,
        "predicted_condition": "Cloudy",
        "predicted_wind_direction": "NE"
    }