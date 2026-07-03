from django.shortcuts import render
from .services import get_weather


def home(request):
    city = request.GET.get("city", "Kigali")

    weather = get_weather(city)

    print("VIEW WEATHER:", weather)  # DEBUG

    return render(request, "dashboard/home.html", {
        "weather": weather
    })