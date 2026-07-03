import requests
from django.conf import settings

def get_weather(city="London"):
    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={settings.OWN_API_KEY}&units=metric"
    )

    response = requests.get(url)
    data = response.json()

    if response.status_code != 200:
        print("API ERROR:", data)
        return None

    return {
        "city": data["name"],
        "temperature": data["main"]["temp"],
        "feels_like": data["main"]["feels_like"],
        "weather_desc": data["weather"][0]["description"],
        "humidity": data["main"]["humidity"],
        "pressure": data["main"]["pressure"],
        "wind_speed": data["wind"]["speed"],
        "cloud_cover": data.get("clouds", {}).get("all", 0),
        "precipitation": 0
    }