from django.urls import path,include
from firstApp import views
urlpatterns = [
    path('', views.index, name='index'),
    path('video1',views.video1,name='video1'),
    path('predictImage',views.predictImage,name='predictImage'),
    path('display',views.display,name='display'),
    path('insert',views.insert,name='insert')
]
