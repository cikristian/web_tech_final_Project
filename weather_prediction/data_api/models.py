from django.db import models

class WeatherReading(models.Model):
    city = models.CharField(max_length=100)
    temperature = models.FloatField()
    feels_like = models.FloatField()
    humidity = models.FloatField()
    pressure = models.FloatField()
    weather_desc = models.CharField(max_length=100)
    wind_speed = models.FloatField()
    precipitation = models.FloatField(default=0.0)
    cloud_cover = models.FloatField(default=0.0)
    fetched_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.city} - {self.temperature}"