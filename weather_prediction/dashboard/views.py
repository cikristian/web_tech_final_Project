from django.shortcuts import render
from data_api.services import get_weather

def home(request):
    city = request.GET.get("city", "London")

    weather = get_weather(city)

    print("VIEW WEATHER:", weather)

    return render(request, "dashboard/home.html", {
        "weather": weather
    })