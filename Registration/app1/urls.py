from django.urls import path
from . import views

urlpatterns = [
    path('home/', views.HomePage, name='home'),
    path('signup/', views.SignupPage, name='signup'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('Image/', views.image_analysis_view, name='image_analysis_view'),
    path('Video/', views.video_analysis, name='video_analysis'),
    path('image-upload/', views.image_upload_view, name='image_upload_view'),
    path('image-analysis/', views.image_analysis_view, name='image_analysis'),
    
]

