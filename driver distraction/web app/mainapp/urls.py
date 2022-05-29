from django.urls import path,include
from mainapp import views

urlpatterns = [
    path('', views.index, name='index'),
    path('predictImage',views.predictImage,name='predictImage'),
    path('video1',views.video1,name='video1'),
    path('display',views.display,name='display'),
    path('insert',views.insert,name='insert'),
    path('query',views.query,name='query')
]
