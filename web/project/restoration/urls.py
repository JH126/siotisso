from django.urls import path
from . import views

urlpatterns = [
    path('results/', views.results, name='results'),
    path('', views.index, name='index'),
    path('check/', views.check, name='check'),
    ]
