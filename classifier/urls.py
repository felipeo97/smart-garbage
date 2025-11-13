# classifier/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'), 
    path('api/classify/', views.classify_image, name='classify_image'),
]
