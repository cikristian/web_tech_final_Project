from django.urls import path
from .views import prediction_view

urlpatterns = [
    path("prediction/", prediction_view, name="prediction"),
]