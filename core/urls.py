from django.urls import path

from . import views, views_video

app_name = 'core'

urlpatterns = [
  # Dashboard
  path('', views.DashboardView.as_view(), name='dashboard'),

  # Projectos
  path('projects/', views.ProjectListView.as_view(), name='project_list'),
  path('projects/create/', views.ProjectCreateView.as_view(), name='project_create'),
  path('projects/<uuid:pk>/', views.ProjectDetailView.as_view(), name='project_detail'),
  path('projects/<uuid:pk>/edit/', views.ProjectUpdateView.as_view(), name='project_edit'),
  path('projects/<uuid:pk>/delete/', views.ProjectDeleteView.as_view(), name='project_delete'),

  # Imagenes
  path('projects/<uuid:project_pk>/upload/', views.UploadImageView.as_view(), name='upload_image'),
  path('projects/<uuid:project_pk>/bulk-upload/', views.BulkUploadImagesView.as_view(), name='bulk_upload_images'),
  path('images/<uuid:pk>/', views.ImageDetailView.as_view(), name='image_detail'),

  # API endpoints
  path('api/images/<uuid:image_pk>/data/', views.ImageDataAPIView.as_view(), name='image_data_api'),
  path('api/images/<uuid:image_pk>/process/', views.ProcessImageDetectionView.as_view(),
       name='process_image_detection'),
  path('api/images/<uuid:image_pk>/aruco-check/', views.ArucoCheckView.as_view(), name='aruco_check'),
  path('api/images/<uuid:image_pk>/manual-measurement/', views.SaveManualMeasurementView.as_view(),
       name='save_manual_measurement'),

  # URL para OBTENER la lista de todas las medidas manuales
  path('api/images/<uuid:image_pk>/manual-measurements/', views.GetManualMeasurementsView.as_view(),
       name='api_get_manual_measurements'),
  path('api/images/<uuid:image_pk>/manual-measurement/<uuid:measurement_id>/',
       views.DeleteManualMeasurementView.as_view(),
       name='api_delete_manual_measurement'),
  path('images/<uuid:image_pk>/preview-pdf/', views.PreviewImagePDFView.as_view(), name='preview_image_pdf'),
  path('images/<uuid:image_pk>/download-excel/', views.DownloadImageExcelView.as_view(), name='download_image_excel'),

  # Reportes
  path('projects/<uuid:project_pk>/generate-report/', views.GenerateReportView.as_view(), name='generate_report'),
  path('reports/<uuid:report_pk>/download/', views.DownloadReportView.as_view(), name='download_report'),
  path('projects/<uuid:project_pk>/video/report/', views_video.VideoSessionsReportView.as_view(),
       name='video_sessions_report'),
  path('projects/<uuid:project_pk>/video/report/pdf/',
       views_video.VideoSessionsPDFReportView.as_view(),
       name='video_sessions_pdf_report'),

  # Vistas adicionales
  path('gallery/', views.GalleryView.as_view(), name='gallery'),
  path('statistics/', views.StatisticsView.as_view(), name='statistics'),
  path('help/', views.HelpView.as_view(), name='help'),
  # Authentication: registration
  path('register/', views.RegisterView.as_view(), name='register'),

  # Perfil de usuario y configuración
  path('profile/', views.UserProfileView.as_view(), name='user_profile'),
  path('settings/', views.UserSettingsView.as_view(), name='user_settings'),
  path('api/update-theme/', views.UpdateThemeView.as_view(), name='update_theme'),
  path('settings/export/', views.ExportUserDataView.as_view(), name='export_user_data'),
  path('settings/delete-account/', views.DeleteAccountView.as_view(), name='delete_account'),

  # Video streaming endpoints
  path('projects/<uuid:project_pk>/video/', views_video.VideoDashboardView.as_view(), name='video_dashboard'),
  path('projects/<uuid:project_pk>/video/start/', views_video.StartVideoSessionView.as_view(),
       name='start_video_session'),
  path('projects/<uuid:project_pk>/video/<uuid:session_pk>/', views_video.VideoStreamView.as_view(),
       name='video_stream'),
  path('projects/<uuid:project_pk>/video/<uuid:session_pk>/feed/', views_video.VideoFeedView.as_view(),
       name='video_feed'),
  path('projects/<uuid:project_pk>/video/<uuid:session_pk>/stop/', views_video.StopVideoSessionView.as_view(),
       name='stop_video_session'),
  path('projects/<uuid:project_pk>/video/<uuid:session_pk>/detail/', views_video.VideoSessionDetailView.as_view(),
       name='video_session_detail'),

  # Video API endpoints
  path('api/video/<uuid:project_pk>/<uuid:session_pk>/check-aruco/', views_video.CheckArucoStatusView.as_view(),
       name='check_aruco_status'),
  path('api/video/<uuid:project_pk>/<uuid:session_pk>/click/', views_video.VideoClickView.as_view(),
       name='video_click'),
  path('api/video/<uuid:project_pk>/<uuid:session_pk>/save-measurement/',
       views_video.SaveVideoMeasurementView.as_view(),
       name='save_video_measurement'),
  path('api/video/<uuid:project_pk>/<uuid:session_pk>/save-all/', views_video.SaveAllMeasurementsView.as_view(),
       name='save_all_measurements'),
  path('api/video/<uuid:project_pk>/<uuid:session_pk>/calibrate/', views_video.CalibrateVideoView.as_view(),
       name='calibrate_video'),
  path('api/video/<uuid:project_pk>/<uuid:session_pk>/calibrate-manual/',
       views_video.CalibrateVideoManualView.as_view(),
       name='calibrate_video_manual'),
  path('api/video/<uuid:project_pk>/<uuid:session_pk>/auto-calibrate/',
       views_video.AutoCalibrateWithFingersView.as_view(),
       name='auto_calibrate_video'),
  path('api/video/<uuid:project_pk>/<uuid:session_pk>/gesture-mode/', views_video.SetGestureModeView.as_view(),
       name='set_gesture_mode'),

  path('api/video/<uuid:project_pk>/<uuid:session_pk>/gesture-data/', views_video.GetGestureDataView.as_view(),
       name='get_gesture_data'),
  path('api/video/<uuid:project_pk>/<uuid:session_pk>/clear-measurements/',
       views_video.ClearVideoMeasurementsView.as_view(),
       name='clear_video_measurements'),
  path('api/video/<uuid:project_pk>/<uuid:session_pk>/export/', views_video.VideoMeasurementsExportView.as_view(),
       name='video_measurements_export'),
  path('api/video/<uuid:project_pk>/<uuid:session_pk>/confirm-calibration/',
       views_video.ConfirmCalibrationView.as_view(),
       name='confirm_calibration'),

  # Sistema de medición en tiempo real
  path('measurement/stream/', views_video.VideoStreamView.as_view(), name='video_stream'),
  path('measurement/api/', views_video.MeasurementAPIView.as_view(), name='measurement_api'),
]
