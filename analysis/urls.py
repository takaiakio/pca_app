# urls.py
from django.urls import path
from .views import pca_analysis

urlpatterns = [
    path('pca_analysis/', pca_analysis, name='pca_analysis'),
]
