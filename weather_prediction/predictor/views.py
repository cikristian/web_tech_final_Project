from django.shortcuts import render
from data_api.models import WeatherReading
from .services import predict_next_day

def prediction_view(request):
    readings = WeatherReading.objects.order_by("-id")[:8]

    if len(readings) < 2:
        return render(request, "prediction/prediction.html", {
            "result": None,
            "error": "Not enough weather data"
        })

    latest = readings[0]
    previous = list(reversed(readings[1:]))

    result = predict_next_day(latest, previous)

    return render(request, "prediction/prediction.html", {
        "result": result
    })