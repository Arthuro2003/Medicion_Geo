import base64
import json
import math
import os
import time

import cv2
import numpy as np
import traceback
from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.core.files.base import ContentFile
from django.http import HttpResponse
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404
from django.shortcuts import render, redirect
from django.utils import timezone
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.views.generic import TemplateView
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image

from .forms import VideoSessionForm
from .models import Project, VideoSession, VideoMeasurement, UserPreferences
from .utils.real_time_measurement import RealTimeMeasurement
from .utils.video_processing import VideoStreamHandler

video_handler = VideoStreamHandler()
real_time_measurer = RealTimeMeasurement()

calibration_watch = {}
CALIBRATION_MISS_THRESHOLD = 5


class CalibrateVideoManualView(LoginRequiredMixin, View):
  http_method_names = ['post']
  csrf_exempt = True

  def post(self, request, project_pk, session_pk):
    project = get_object_or_404(Project, pk=project_pk, created_by=request.user)
    video_session = get_object_or_404(VideoSession, pk=session_pk, project=project)

    if not video_session.is_active:
      return JsonResponse({'error': 'Session not active'}, status=400)

    try:
      data = json.loads(request.body)
      aruco_size_cm = float(data.get('aruco_size_cm', 5.0))

      video_session.calibration_data = {
        'aruco_size_cm': aruco_size_cm,
        'calibration_type': 'aruco',
        'calibrated_at': timezone.now().isoformat()
      }
      video_session.is_calibrated = True
      video_session.save()

      video_handler.set_calibration_data({'aruco_size_cm': aruco_size_cm})

      return JsonResponse({
        'success': True,
        'calibration': {
          'aruco_size_cm': aruco_size_cm
        }
      })

    except Exception as e:
      return JsonResponse({'error': str(e)}, status=500)


class MeasurementAPIView(TemplateView):
  measurer = None

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    if not MeasurementAPIView.measurer:
      MeasurementAPIView.measurer = RealTimeMeasurement()

  def generate_frames(self, session_pk=None):
    if not video_handler.is_streaming:
      video_handler.start_stream(0)

    while video_handler.is_streaming:
      frame = video_handler.processor.capture_frame()
      if frame is None:
        break

      _, buffer = cv2.imencode('.jpg', frame)
      frame_bytes = buffer.tobytes()
      yield (b'--frame\r\n'
             b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

  def get(self, request, *args, **kwargs):
    session_pk = request.GET.get('session_pk')
    if request.headers.get('accept') == 'text/event-stream':
      return StreamingHttpResponse(
        self.generate_frames(session_pk=session_pk),
        content_type='multipart/x-mixed-replace; boundary=frame'
      )
    return super().get(request, *args, **kwargs)

  @method_decorator(csrf_exempt, name='dispatch')
  def post(self, request, *args, **kwargs):
    action = request.POST.get('action')

    if action == 'calibrate_color':
      try:
        data = json.loads(request.body)
        roi = data.get('roi')
        frame_data = data.get('frame')

        frame_bytes = base64.b64decode(frame_data.split(',')[1])
        frame_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)

        self.measurer.calibrate_color(frame, roi)

        return JsonResponse({'status': 'success'})
      except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)})

    elif action == 'get_measurement':
      try:
        frame_data = json.loads(request.body).get('frame')
        frame_bytes = base64.b64decode(frame_data.split(',')[1])
        frame_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)

        measurement_data = self.measurer.get_measurement_data(frame)

        return JsonResponse({
          'status': 'success',
          'data': measurement_data
        })
      except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)})

    return JsonResponse({'status': 'error', 'message': 'Invalid action'})


class VideoDashboardView(LoginRequiredMixin, View):
  http_method_names = ['get']

  def get(self, request, project_pk):
    project = get_object_or_404(Project, pk=project_pk, created_by=request.user)
    video_sessions = project.video_sessions.all()[:10]

    context = {
      'project': project,
      'video_sessions': video_sessions,
      'active_session': project.video_sessions.filter(is_active=True).first()
    }
    return render(request, 'core/video_dashboard.html', context)


class StartVideoSessionView(LoginRequiredMixin, View):

  def get(self, request, project_pk):
    project = get_object_or_404(Project, pk=project_pk, created_by=request.user)
    form = VideoSessionForm()

    context = {
      'project': project,
      'form': form
    }
    return render(request, 'core/start_video_session.html', context)

  def post(self, request, project_pk):
    project = get_object_or_404(Project, pk=project_pk, created_by=request.user)

    form = VideoSessionForm(request.POST)
    if form.is_valid():
      active_session = project.video_sessions.filter(is_active=True).first()
      if active_session:
        messages.warning(request, 'Ya hay una sesión de video activa. Termínala antes de iniciar una nueva.')
        return redirect('core:video_dashboard', project_pk=project.pk)

      video_session = form.save(commit=False)
      video_session.project = project
      video_session.is_active = True
      video_session.save()

      if video_handler.start_stream(0):
        messages.success(request, f'Sesión de video "{video_session.name}" iniciada correctamente.')
        return redirect('core:video_stream', project_pk=project.pk, session_pk=video_session.pk)
      else:
        video_session.delete()
        messages.error(request, 'No se pudo acceder a la cámara. Verifica que esté conectada y disponible.')
        return redirect('core:video_dashboard', project_pk=project.pk)

    context = {
      'project': project,
      'form': form
    }
    return render(request, 'core/start_video_session.html', context)


class VideoStreamView(TemplateView):
  template_name = 'core/video_stream.html'

  def get_context_data(self, **kwargs):
    context = super().get_context_data(**kwargs)
    project = get_object_or_404(Project, pk=kwargs.get('project_pk'), created_by=self.request.user)
    video_session = get_object_or_404(VideoSession, pk=kwargs.get('session_pk'), project=project)
    preferences = UserPreferences.get_or_create_for_user(self.request.user)

    if video_session.is_calibrated and video_session.calibration_data:
      video_handler.set_calibration_data(video_session.calibration_data)

    context = {
      'project': project,
      'video_session': video_session,
      'preferences': preferences
    }
    return context  

  def get(self, request, *args, **kwargs):
    project = get_object_or_404(Project, pk=kwargs.get('project_pk'), created_by=request.user)
    video_session = get_object_or_404(VideoSession, pk=kwargs.get('session_pk'), project=project)

    if not video_session.is_active:
      messages.error(request, 'Esta sesión de video no está activa.')
      return redirect('core:video_dashboard',
                      project_pk=project.pk)  

    context = self.get_context_data(**kwargs)
    return self.render_to_response(context)  


class VideoFeedView(LoginRequiredMixin, View):
  http_method_names = ['get']

  def get(self, request, project_pk, session_pk):
    project = get_object_or_404(Project, pk=project_pk)
    video_session = get_object_or_404(VideoSession, pk=session_pk, project=project)

    if not video_session.is_active:
      return HttpResponse("Session not active", status=400)

    def generate_frames():
      while video_session.is_active and video_handler.is_streaming:
        frame_bytes = video_handler.get_frame_bytes()
        if frame_bytes:
          yield (b'--frame\r\n'
                 b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
          time.sleep(0.1)

    return StreamingHttpResponse(
      generate_frames(),
      content_type='multipart/x-mixed-replace; boundary=frame'
    )


class CheckArucoStatusView(LoginRequiredMixin, View):
  http_method_names = ['get']
  csrf_exempt = True

  def get(self, request, project_pk, session_pk):
    try:
      video_session = get_object_or_404(VideoSession, pk=session_pk)

      aruco_detected = False
      measurements = []

      try:
        measurements = video_handler.get_measurements()
      except Exception:
        measurements = []

      if measurements and len(measurements) > 0:
        aruco_detected = True

      if not aruco_detected:
        try:
          if hasattr(video_handler, 'measurement_system') and hasattr(video_handler.measurement_system,
                                                                      'ratio_history'):
            ratio_history = list(video_handler.measurement_system.ratio_history)
            if len(ratio_history) > 0 and any((r is not None and r > 0) for r in ratio_history):
              aruco_detected = True
        except Exception:
          pass

      if not aruco_detected and hasattr(video_handler, 'processor'):
        try:
          proc = video_handler.processor
          if hasattr(proc, 'is_aruco_detected'):
            aruco_detected = bool(proc.is_aruco_detected())
          else:
            if hasattr(proc, 'measurement_system') and hasattr(proc.measurement_system, 'ratio_history'):
              rh = list(proc.measurement_system.ratio_history)
              if len(rh) > 0 and any((r is not None and r > 0) for r in rh):
                aruco_detected = True
        except Exception:
          pass

      key = str(session_pk)
      if key not in calibration_watch:
        calibration_watch[key] = {
          'miss_count': 0,
          'last_seen': timezone.now() if aruco_detected else None
        }

      if aruco_detected:
        calibration_watch[key]['miss_count'] = 0
        calibration_watch[key]['last_seen'] = timezone.now()
      else:
        calibration_watch[key]['miss_count'] += 1

      calibration_lost = False
      if video_session.is_calibrated and calibration_watch[key]['miss_count'] >= CALIBRATION_MISS_THRESHOLD:
        try:
          video_session.is_calibrated = False
          video_session.save()
          calibration_lost = True
        except Exception:
          pass

      def sanitize_value(v):
        try:
          import numpy as _np
          if isinstance(v, (_np.floating, _np.integer)):
            return float(v)
          if isinstance(v, _np.ndarray):
            return v.tolist()
        except Exception:
          pass

        try:
          from collections import deque as _deque
          if isinstance(v, _deque):
            return list(v)
        except Exception:
          pass

        if isinstance(v, list):
          return [sanitize_value(x) for x in v]
        if isinstance(v, dict):
          return {str(k): sanitize_value(val) for k, val in v.items()}

        if isinstance(v, (int, float, str, bool)) or v is None:
          return v

        try:
          return float(v)
        except Exception:
          try:
            return int(v)
          except Exception:
            return str(v)

      sanitized_measurements = []
      try:
        for m in measurements:
          if not isinstance(m, dict):
            sanitized_measurements.append(sanitize_value(m))
            continue
          sanitized = {str(k): sanitize_value(v) for k, v in m.items()}
          sanitized_measurements.append(sanitized)
      except Exception:
        sanitized_measurements = []

      return JsonResponse({
        'aruco_detected': aruco_detected,
        'is_calibrated': video_session.is_calibrated,
        'calibration_data': video_session.calibration_data,
        'has_measurements': len(sanitized_measurements) > 0,
        'measurements': sanitized_measurements,
        'status': 'calibrated' if video_session.is_calibrated else ('detected' if aruco_detected else 'waiting'),
        'calibration_lost': calibration_lost
      })
    except Exception as e:
      import traceback
      print("Error en check_aruco_status:", str(e))
      print(traceback.format_exc())
      return JsonResponse({
        'error': str(e),
        'aruco_detected': False,
        'is_calibrated': False,
        'calibration_data': {}
      })


class VideoClickView(LoginRequiredMixin, View):
  http_method_names = ['post']
  csrf_exempt = True

  def post(self, request, project_pk, session_pk):
    project = get_object_or_404(Project, pk=project_pk)
    video_session = get_object_or_404(VideoSession, pk=session_pk, project=project)

    if not video_session.is_active:
      return JsonResponse({'error': 'Session not active'}, status=400)

    try:
      data = json.loads(request.body)
      x = int(data.get('x', 0))
      y = int(data.get('y', 0))

      video_handler.process_click(x, y)

      points = video_handler.processor.measurement_points

      response_data = {
        'success': True,
        'points': points,
        'measurement_count': len(points)
      }

      if len(points) == 2:
        from .utils.geometry import GeometryCalculator
        p1 = {'x': points[0][0], 'y': points[0][1]}
        p2 = {'x': points[1][0], 'y': points[1][1]}

        pixel_dist = GeometryCalculator().calculate_distance(p1, p2) if hasattr(GeometryCalculator,
                                                                                'calculate_distance') else None
        response_data['distance'] = pixel_dist

        try:
          geom = video_handler.processor.geometry_calculator
          if geom and geom.is_calibrated():
            real_distance = geom.pixel_to_cm(p1, p2)
            response_data['real_distance'] = float(real_distance)
            response_data['unit'] = geom.unit if hasattr(geom, 'unit') else 'cm'
        except Exception:
          cal_data = video_session.calibration_data or {}
          if cal_data.get('pixels_per_unit'):
            try:
              real_distance = pixel_dist / float(cal_data['pixels_per_unit'])
              response_data['real_distance'] = real_distance
              response_data['unit'] = cal_data.get('unit', 'units')
            except Exception:
              pass

      return JsonResponse(response_data)

    except Exception as e:
      return JsonResponse({'error': str(e)}, status=500)


class SaveVideoMeasurementView(LoginRequiredMixin, View):
  http_method_names = ['post']
  csrf_exempt = True

  def post(self, request, project_pk, session_pk):
    project = get_object_or_404(Project, pk=project_pk, created_by=request.user)
    video_session = get_object_or_404(VideoSession, pk=session_pk, project=project)

    if not video_session.is_active:
      return JsonResponse({'error': 'Session not active'}, status=400)

    try:
      data = json.loads(request.body)

      current_measurements = video_handler.get_measurements()

      saved_measurements = []
      for i, obj_data in enumerate(current_measurements):
        shape_type = obj_data.get('shape_type', 'unknown')
        area = obj_data.get('area_cm2', 0)

        if shape_type == 'Circulo':
          measurement_type = 'area'
          value_real = area
          perimeter = 2 * math.pi * obj_data.get('radius_cm', 0)
        elif shape_type == 'Triangulo':
          measurement_type = 'area'
          value_real = area
          sides = obj_data.get('sides_cm', [])
          perimeter = sum(sides) if sides else 0
        else:
          measurement_type = 'area'
          value_real = area
          width = obj_data.get('width_cm', 0)
          height = obj_data.get('height_cm', 0)
          perimeter = 2 * (width + height)

        calibration_data = video_session.calibration_data or {}
        aruco_size_cm = calibration_data.get('aruco_size_cm', 1.0)

        measurement = VideoMeasurement.objects.create(
          video_session=video_session,
          name=f'{shape_type} {i + 1} - {timezone.now().strftime("%H:%M:%S")}',
          measurement_type=measurement_type,
          shape_type=shape_type,
          value_pixels=area * (aruco_size_cm ** 2),
          value_real=value_real,
          unit='cm²',
          width=obj_data.get('width_cm'),
          height=obj_data.get('height_cm'),
          area=area,
          perimeter=perimeter,
          points=obj_data.get('points', []),
          frame_data={
            'timestamp': time.time(),
            'shape_data': obj_data
          },
          timestamp=time.time()
        )

        saved_measurements.append({
          'id': str(measurement.id),
          'name': measurement.name,
          'value_display': measurement.value_display,
          'created_at': measurement.created_at.isoformat(),
          'shape_type': shape_type,
          'measurements': obj_data
        })

      manual_points = video_handler.measurement_points
      if len(manual_points) == 2:
        p1 = np.array(manual_points[0])
        p2 = np.array(manual_points[1])
        pixel_distance = np.linalg.norm(p2 - p1)

        manual_measurement = VideoMeasurement.objects.create(
          video_session=video_session,
          name=f'Medición Manual - {timezone.now().strftime("%H:%M:%S")}',
          measurement_type='distance',
          value_pixels=pixel_distance,
          unit='px',
          points=manual_points,
          frame_data={'timestamp': time.time()},
          timestamp=time.time()
        )

        saved_measurements.append({
          'id': str(manual_measurement.id),
          'name': manual_measurement.name,
          'value_display': manual_measurement.value_display,
          'created_at': manual_measurement.created_at.isoformat(),
          'type': 'manual_distance',
          'points': manual_points
        })

      return JsonResponse({
        'success': True,
        'measurements': saved_measurements
      })

    except Exception as e:
      return JsonResponse({'error': str(e)}, status=500)


class SaveAllMeasurementsView(LoginRequiredMixin, View):
  http_method_names = ['post']
  csrf_exempt = True

  def post(self, request, project_pk, session_pk):
    project = get_object_or_404(Project, pk=project_pk, created_by=request.user)
    video_session = get_object_or_404(VideoSession, pk=session_pk, project=project)

    if not video_session.is_active:
      return JsonResponse({'error': 'La sesión no está activa'}, status=400)

    try:
      frame_bytes = video_handler.get_frame_bytes()

      if frame_bytes and not video_session.snapshot:
        file_name = f'snapshot_session_{video_session.id}.jpg'

        image_file = ContentFile(frame_bytes, name=file_name)

        video_session.snapshot = image_file

        video_session.save(update_fields=['snapshot'])

      current_measurements = video_handler.get_measurements()
      saved_measurements = []

      def sanitize_value(v):
        try:
          if isinstance(v, (np.floating, np.integer)):
            return float(v)
          if isinstance(v, np.ndarray):
            return v.tolist()
        except ImportError:
          pass
        except Exception:
          pass

        if isinstance(v, list):
          return [sanitize_value(x) for x in v]
        if isinstance(v, dict):
          return {str(k): sanitize_value(val) for k, val in v.items()}
        if isinstance(v, (int, float, str, bool)) or v is None:
          return v
        try:
          return float(v)
        except (ValueError, TypeError):
          try:
            return str(v)
          except Exception:
            return None

      for i, obj_data in enumerate(current_measurements):
        try:
          if not isinstance(obj_data, dict):
            continue

          obj = {str(k): sanitize_value(v) for k, v in obj_data.items()}

          shape_type = obj.get('shape_type') or obj.get('type') or 'Objeto'
          area = float(obj.get('area_cm2') or obj.get('area') or 0.0)
          width = float(obj.get('width_cm') or obj.get('width') or 0.0)
          height = float(obj.get('height_cm') or obj.get('height') or 0.0)
          radius = float(obj.get('radius_cm') or obj.get('radius') or 0.0)
          diameter = float(obj.get('diameter_cm') or obj.get('diameter') or (radius * 2) or 0.0)
          perimeter = float(obj.get('perimeter_cm') or obj.get('perimeter') or 0.0)
          sides = obj.get('sides_cm') or obj.get('sides') or []

          if isinstance(sides, list):
            sides = [float(x) if x is not None else 0.0 for x in sides]
          else:
            sides = []

          calibration_data = video_session.calibration_data or {}
          aruco_size_cm = float(calibration_data.get('aruco_size_cm', 1.0) or 1.0)
          value_pixels = float(area * (aruco_size_cm ** 2)) if area > 0 else 0.0

          vm = VideoMeasurement.objects.create(
            video_session=video_session,
            name=f"{shape_type} {i + 1} - {timezone.now().strftime('%H:%M:%S')}",
            measurement_type='area',
            value_pixels=value_pixels,
            value_real=area,
            unit='cm²' if area > 0 else 'px',
            shape_type=shape_type,
            width=(width if width > 0 else None),
            height=(height if height > 0 else None),
            area=(area if area > 0 else None),
            perimeter=(perimeter if perimeter > 0 else None),
            radius=(radius if radius > 0 else None),
            diameter=(diameter if diameter > 0 else None),
            side_a=(sides[0] if len(sides) > 0 else None),
            side_b=(sides[1] if len(sides) > 1 else None),
            side_c=(sides[2] if len(sides) > 2 else None),
            object_id=(int(obj.get('object_id')) if obj.get('object_id') is not None else None),
            is_active=bool(obj.get('is_active', True)),
            points=sanitize_value(obj.get('points') or []),
            frame_data={'timestamp': time.time(), 'shape_data': obj},
            timestamp=time.time()
          )

          saved_measurements.append({
            'id': str(vm.id),
            'name': vm.name
          })

        except Exception as e_obj:
          print(f"Error guardando objeto {i}: {e_obj}")
          continue

      video_handler.clear_measurements()

      return JsonResponse({
        'success': True,
        'message': f'Se guardaron {len(saved_measurements)} mediciones correctamente.',
        'measurements': saved_measurements
      })
    except Exception as e:
      print('Error en save_all_measurements:', str(e))
      print(traceback.format_exc())
      return JsonResponse({'error': f'Error interno del servidor: {str(e)}'}, status=500)


class CalibrateVideoView(LoginRequiredMixin, View):
  http_method_names = ['post']
  csrf_exempt = True

  def post(self, request, project_pk, session_pk):
    project = get_object_or_404(Project, pk=project_pk, created_by=request.user)
    video_session = get_object_or_404(VideoSession, pk=session_pk, project=project)

    if not video_session.is_active:
      return JsonResponse({'error': 'Session not active'}, status=400)

    try:
      data = json.loads(request.body)
      pixel_distance = data.get('pixel_distance')
      real_distance = data.get('real_distance')
      unit = data.get('unit', 'cm')

      if not pixel_distance or not real_distance:
        return JsonResponse({'error': 'Missing pixel_distance or real_distance'}, status=400)

      pixels_per_unit = pixel_distance / real_distance

      video_session.calibration_data = {
        'pixels_per_unit': pixels_per_unit,
        'unit': unit,
        'reference_pixel_distance': pixel_distance,
        'reference_real_distance': real_distance,
        'calibrated_at': timezone.now().isoformat()
      }
      video_session.is_calibrated = True
      video_session.save()

      video_handler.set_calibration_data({'pixels_per_unit': pixels_per_unit})

      return JsonResponse({
        'success': True,
        'calibration': {
          'pixels_per_unit': pixels_per_unit,
          'unit': unit
        }
      })

    except Exception as e:
      return JsonResponse({'error': str(e)}, status=500)


class AutoCalibrateWithFingersView(LoginRequiredMixin, View):
  http_method_names = ['post']
  csrf_exempt = True

  def post(self, request, project_pk, session_pk):
    project = get_object_or_404(Project, pk=project_pk, created_by=request.user)
    video_session = get_object_or_404(VideoSession, pk=session_pk, project=project)

    if not video_session.is_active:
      return JsonResponse({'error': 'Session not active'}, status=400)

    try:
      data = json.loads(request.body)
      known_distance_cm = data.get('known_distance_cm')

      if not known_distance_cm:
        return JsonResponse({'error': 'Missing known_distance_cm'}, status=400)

      frame = video_handler.processor.capture_frame()
      if frame is None:
        return JsonResponse({'error': 'Could not capture frame'}, status=500)

      calibration_result = video_handler.processor.auto_calibrate_with_fingers(frame, known_distance_cm)

      if calibration_result.get('calibration_success'):
        pixels_per_cm = calibration_result['pixels_per_cm']
        pixel_distance = calibration_result['calibration_distance']

        video_session.calibration_data = {
          'pixels_per_unit': pixels_per_cm,
          'unit': 'cm',
          'reference_pixel_distance': pixel_distance,
          'reference_real_distance': known_distance_cm,
          'calibrated_at': timezone.now().isoformat(),
          'auto_calibrated': True,
          'calibration_hand_ref_distance': calibration_result.get('calibration_hand_ref_distance')
        }
        video_session.is_calibrated = True
        video_session.save()

        return JsonResponse({
          'success': True,
          'calibration': {
            'pixels_per_cm': pixels_per_cm,
            'pixel_distance': pixel_distance,
            'real_distance': known_distance_cm,
            'unit': 'cm'
          }
        })
      else:
        return JsonResponse({
          'success': False,
          'message': 'No finger span detected. Please spread your thumb and index finger.'
        })

    except Exception as e:
      return JsonResponse({'error': str(e)}, status=500)


class SetGestureModeView(LoginRequiredMixin, View):
  http_method_names = ['post']
  csrf_exempt = True

  def post(self, request, project_pk, session_pk):
    project = get_object_or_404(Project, pk=project_pk)
    video_session = get_object_or_404(VideoSession, pk=session_pk, project=project)

    if not video_session.is_active:
      return JsonResponse({'error': 'Session not active'}, status=400)

    try:
      data = json.loads(request.body)
      gesture_mode = data.get('gesture_mode', True)
      calibration_mode = data.get('calibration_mode', False)
      measurement_mode = data.get('measurement_mode', False)

      video_handler.processor.set_gesture_mode(gesture_mode)
      video_handler.processor.set_calibration_mode(calibration_mode)
      video_handler.processor.set_measurement_mode(measurement_mode)

      return JsonResponse({
        'success': True,
        'modes': {
          'gesture_mode': gesture_mode,
          'calibration_mode': calibration_mode,
          'measurement_mode': measurement_mode
        }
      })

    except Exception as e:
      return JsonResponse({'error': str(e)}, status=500)


class GetGestureDataView(LoginRequiredMixin, View):
  http_method_names = ['get']
  csrf_exempt = True

  def get(self, request, project_pk, session_pk):
    project = get_object_or_404(Project, pk=project_pk)
    video_session = get_object_or_404(VideoSession, pk=session_pk, project=project)

    if not video_session.is_active:
      return JsonResponse({'error': 'Session not active'}, status=400)

    try:
      measurement_points = video_handler.processor.get_gesture_measurement_points()
      calibration_distance = video_handler.processor.get_gesture_calibration_distance()

      auto_calibration_info = video_handler.processor.get_auto_calibration_info()

      return JsonResponse({
        'success': True,
        'gesture_data': {
          'measurement_points': measurement_points,
          'calibration_distance': calibration_distance,
          'auto_calibration': auto_calibration_info
        }
      })

    except Exception as e:
      return JsonResponse({'error': str(e)}, status=500)


class ConfirmCalibrationView(LoginRequiredMixin, View):
  http_method_names = ['post']
  csrf_exempt = True

  def post(self, request, project_pk, session_pk):
    project = get_object_or_404(Project, pk=project_pk)
    video_session = get_object_or_404(VideoSession, pk=session_pk, project=project)

    if not video_session.is_active:
      return JsonResponse({'error': 'Session not active'}, status=400)

    try:
      pixels_per_cm = None
      try:
        ms = video_handler.measurement_system
        ratios = list(ms.ratio_history) if hasattr(ms, 'ratio_history') else []
        if ratios:
          pixels_per_cm = float(sorted([r for r in ratios if r and r > 0])[len(ratios) // 2])
      except Exception:
        pixels_per_cm = None

      if not pixels_per_cm:
        return JsonResponse({'success': False, 'message': 'No calibration ratio available'}, status=400)

      video_session.calibration_data = {
        'pixels_per_unit': pixels_per_cm,
        'unit': 'cm',
        'pixels_per_cm': pixels_per_cm,
        'aruco_size_cm': ms.aruco_size_cm if hasattr(ms, 'aruco_size_cm') else 5.0,
        'calibrated_at': timezone.now().isoformat()
      }
      video_session.is_calibrated = True
      video_session.save()

      video_handler.set_calibration_data(
        {'pixels_per_unit': pixels_per_cm, 'aruco_size_cm': video_session.calibration_data.get('aruco_size_cm')})

      return JsonResponse({'success': True, 'calibration': video_session.calibration_data})
    except Exception as e:
      return JsonResponse({'error': str(e)}, status=500)


class StopVideoSessionView(LoginRequiredMixin, View):
  http_method_names = ['get']

  def get(self, request, project_pk, session_pk):
    project = get_object_or_404(Project, pk=project_pk, created_by=request.user)
    video_session = get_object_or_404(VideoSession, pk=session_pk, project=project)

    if video_session.is_active:
      video_handler.stop_stream()

      video_session.is_active = False
      video_session.ended_at = timezone.now()
      video_session.save()

      messages.success(request, f'Sesión de video "{video_session.name}" terminada correctamente.')
    else:
      messages.warning(request, 'Esta sesión ya estaba terminada.')

    return redirect('core:video_dashboard', project_pk=project.pk)


class VideoSessionDetailView(LoginRequiredMixin, View):
  http_method_names = ['get']

  def get(self, request, project_pk, session_pk):
    project = get_object_or_404(Project, pk=project_pk, created_by=request.user)
    video_session = get_object_or_404(VideoSession, pk=session_pk, project=project)
    measurements = video_session.measurements.all()

    context = {
      'project': project,
      'video_session': video_session,
      'measurements': measurements
    }
    return render(request, 'core/video_session_detail.html', context)


class ClearVideoMeasurementsView(LoginRequiredMixin, View):
  http_method_names = ['post']
  csrf_exempt = True

  def post(self, request, project_pk, session_pk):
    project = get_object_or_404(Project, pk=project_pk)
    video_session = get_object_or_404(VideoSession, pk=session_pk, project=project)

    if not video_session.is_active:
      return JsonResponse({'error': 'Session not active'}, status=400)

    try:
      video_handler.clear_measurements()

      return JsonResponse({
        'success': True,
        'message': 'Mediciones y señales limpiadas correctamente'
      })
    except Exception as e:
      return JsonResponse({'error': str(e)}, status=500)


class VideoMeasurementsExportView(LoginRequiredMixin, View):
  http_method_names = ['get']

  def get(self, request, project_pk, session_pk):
    project = get_object_or_404(Project, pk=project_pk, created_by=request.user)
    video_session = get_object_or_404(VideoSession, pk=session_pk, project=project)

    format_type = request.GET.get('format', 'json')

    if format_type == 'json':
      measurements_data = []
      for measurement in video_session.measurements.all():
        measurements_data.append({
          'name': measurement.name,
          'type': measurement.measurement_type,
          'shape_type': measurement.shape_type,
          'width': measurement.width,
          'height': measurement.height,
          'area': measurement.area,
          'perimeter': measurement.perimeter,
          'value_pixels': measurement.value_pixels,
          'value_real': measurement.value_real,
          'unit': measurement.unit,
          'points': measurement.points,
          'timestamp': measurement.timestamp,
          'created_at': measurement.created_at.isoformat()
        })

      response_data = {
        'session': {
          'name': video_session.name,
          'started_at': video_session.started_at.isoformat(),
          'ended_at': video_session.ended_at.isoformat() if video_session.ended_at else None,
          'is_calibrated': video_session.is_calibrated
        },
        'measurements': measurements_data
      }

      response = JsonResponse(response_data, json_dumps_params={'indent': 2})
      response['Content-Disposition'] = f'attachment; filename="video_measurements_{video_session.id}.json"'
      return response

    else:
      return JsonResponse({'error': 'Unsupported format'}, status=400)


class VideoSessionsReportView(LoginRequiredMixin, View):
  http_method_names = ['get']

  def get(self, request, project_pk):
    project = get_object_or_404(Project, pk=project_pk, created_by=request.user)
    video_sessions = VideoSession.objects.filter(project=project).prefetch_related('measurements').order_by(
      '-started_at')

    format_type = request.GET.get('format', 'json')

    if format_type == 'json':
      sessions_data = []
      for session in video_sessions:
        measurements_data = []
        for measurement in session.measurements.all():
          measurements_data.append({
            'name': measurement.name,
            'measurement_type': measurement.measurement_type,
            'shape_type': measurement.shape_type,
            'value_pixels': measurement.value_pixels,
            'value_real': measurement.value_real,
            'unit': measurement.unit,
            'width': measurement.width,
            'height': measurement.height,
            'area': measurement.area,
            'perimeter': measurement.perimeter,
            'radius': measurement.radius,
            'diameter': measurement.diameter,
            'side_a': measurement.side_a,
            'side_b': measurement.side_b,
            'side_c': measurement.side_c,
            'object_id': measurement.object_id,
            'is_active': measurement.is_active,
            'points': measurement.points,
            'timestamp': measurement.timestamp,
            'created_at': measurement.created_at.isoformat()
          })

        sessions_data.append({
          'id': str(session.id),
          'name': session.name,
          'is_active': session.is_active,
          'started_at': session.started_at.isoformat(),
          'ended_at': session.ended_at.isoformat() if session.ended_at else None,
          'is_calibrated': session.is_calibrated,
          'calibration_data': session.calibration_data,
          'settings': session.settings,
          'duration': session.duration.total_seconds() if session.duration else None,
          'measurements': measurements_data
        })

      response_data = {
        'project': {
          'id': str(project.id),
          'name': project.name,
          'description': project.description
        },
        'video_sessions': sessions_data,
        'generated_at': timezone.now().isoformat(),
        'total_sessions': len(sessions_data),
        'total_measurements': sum(s.measurements.count() for s in video_sessions)
      }

      response = JsonResponse(response_data, json_dumps_params={'indent': 2})
      filename = f'video_sessions_report_{project.id}.json'
      response['Content-Disposition'] = f'attachment; filename="{filename}"'
      return response

    else:
      messages.error(request, 'Formato no soportado para el reporte de sesiones de video.')
      return redirect('core:video_dashboard', project_pk=project.pk)


class VideoSessionsPDFReportView(LoginRequiredMixin, View):
  def get(self, request, project_pk):
    project = get_object_or_404(Project, pk=project_pk, created_by=request.user)
    sessions = VideoSession.objects.filter(project=project).prefetch_related('measurements').order_by('-started_at')

    if not sessions.exists():
      return HttpResponse("No hay sesiones de video registradas para este proyecto.", status=404)

    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="reporte_sesiones_{project.name}.pdf"'

    doc = SimpleDocTemplate(
      response,
      pagesize=A4,
      rightMargin=2 * cm,
      leftMargin=2 * cm,
      topMargin=2 * cm,
      bottomMargin=2 * cm
    )

    styles = getSampleStyleSheet()

    if 'SessionTitle' not in styles:
      styles.add(ParagraphStyle(
        name='SessionTitle',
        fontSize=16,
        textColor=colors.HexColor("#6C63FF"),
        spaceAfter=12,
        leading=18,
        fontName='Helvetica-Bold'
      ))
    if 'SectionTitle' not in styles:
      styles.add(ParagraphStyle(
        name='SectionTitle',
        fontSize=13,
        textColor=colors.HexColor("#6C63FF"),
        spaceAfter=10,
        fontName='Helvetica-Bold'
      ))
    if 'CustomBodyText' not in styles:
      styles.add(ParagraphStyle(
        name='CustomBodyText',
        fontSize=11,
        spaceAfter=4
      ))

    story = []
    story.append(Paragraph("Reporte de Sesiones de Video", styles['Title']))
    story.append(Paragraph(f"Proyecto: <b>{project.name}</b>", styles['CustomBodyText']))
    story.append(Spacer(1, 12))

    for session in sessions:
      story.append(Paragraph(f"Sesión: {session.name}", styles['SessionTitle']))

      duration_display = '—'
      if session.started_at and session.ended_at:
        duration_seconds = (session.ended_at - session.started_at).total_seconds()
        hours = int(duration_seconds // 3600)
        minutes = int((duration_seconds % 3600) // 60)
        seconds = int(duration_seconds % 60)
        duration_display = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
      calibration_status = 'Calibrada' if getattr(session, 'is_calibrated', False) else 'No calibrada'

      details_data = [
        ['Iniciada:', session.started_at.strftime('%d/%m/%Y %H:%M:%S') if session.started_at else '—'],
        ['Terminada:', session.ended_at.strftime('%d/%m/%Y %H:%M:%S') if session.ended_at else '—'],
        ['Duración:', duration_display],
        ['Estado:', calibration_status],
      ]

      if hasattr(session, 'calibration_data') and session.calibration_data:
        calib = session.calibration_data
        details_data.append(['Unidad:', calib.get('unit', '—')])
        pixels_per_unit = calib.get('pixels_per_unit', None)
        pixels_formatted = str(pixels_per_unit) if pixels_per_unit is not None else '—'
        if isinstance(pixels_per_unit, (float, int)):
          pixels_formatted = f"{pixels_per_unit:.2f}"
        details_data.append(['Pixeles por unidad:', pixels_formatted])

      details_table = Table(details_data, colWidths=[5 * cm, 10 * cm])
      details_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6C63FF')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.gray),
        ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke)
      ]))
      story.append(details_table)
      story.append(Spacer(1, 10))

      if session.snapshot and hasattr(session.snapshot, 'path') and os.path.exists(session.snapshot.path):
        try:
          story.append(Paragraph("<b>Captura de la Sesión:</b>", styles['CustomBodyText']))
          max_width = 12 * cm
          img_reader = ImageReader(session.snapshot.path)
          original_width, original_height = img_reader.getSize()
          aspect = original_height / float(original_width)
          snapshot_image = Image(session.snapshot.path, width=max_width, height=(max_width * aspect))
          story.append(snapshot_image)
          story.append(Spacer(1, 10))
        except Exception as e:
          print(f"Error al procesar la imagen de la sesión {session.id}: {e}")

      measurements = session.measurements.all()
      measurements_count = measurements.count()

      if measurements_count > 0:
        story.append(Paragraph(f"Mediciones ({measurements_count})", styles['SectionTitle']))
        for m in measurements:
          m_data = [['Tipo:', m.shape_type or 'N/A'], ['Hora:', m.created_at.strftime('%H:%M:%S')]]
          if m.object_id is not None:
            m_data.append(['Objeto ID:', f'Obj: {m.object_id}'])
          if m.width is not None: m_data.append(['Ancho:', f"{m.width:.2f} cm"])
          if m.height is not None: m_data.append(['Alto:', f"{m.height:.2f} cm"])
          if m.radius is not None: m_data.append(['Radio:', f"{m.radius:.2f} cm"])
          if m.diameter is not None: m_data.append(['Diámetro:', f"{m.diameter:.2f} cm"])
          if m.area is not None: m_data.append(['Área:', f"{m.area:.2f} cm²"])
          if m.perimeter is not None: m_data.append(['Perímetro:', f"{m.perimeter:.2f} cm"])

          if len(m_data) <= 2:
            m_data.append(['Sin mediciones disponibles', '—'])

          m_table = Table(m_data, colWidths=[5 * cm, 10 * cm])
          m_table.setStyle(TableStyle([
            ('BOX', (0, 0), (-1, -1), 0.25, colors.HexColor('#6C63FF')),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('BACKGROUND', (0, 0), (-1, -1), colors.Color(0.98, 0.98, 1)),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.gray)
          ]))
          story.append(m_table)
          story.append(Spacer(1, 8))
      else:
        story.append(Paragraph("No hay mediciones registradas para esta sesión.", styles['CustomBodyText']))
      story.append(PageBreak())

    doc.build(story)
    return response
