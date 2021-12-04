from django.urls import path, re_path
from .views import detect, FileUploadView

urlpatterns = [
    path('face-detection/', detect, name='face-detection'),
    re_path(r'^upload-video/(?P<filename>[^/]+)$', FileUploadView.as_view())
    ]