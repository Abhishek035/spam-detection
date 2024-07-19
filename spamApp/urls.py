from django.urls import path
from spamApp import views

urlpatterns = [
    path('', views.spamApp, name="spamApp"),
    path('result', views.result, name="result")
]
