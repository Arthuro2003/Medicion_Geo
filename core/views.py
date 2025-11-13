import json
import logging
import os
import boto3
from django.conf import settings
import cv2
import numpy as np
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import logout
from django.contrib.auth.mixins import LoginRequiredMixin
from django.core.mail import EmailMessage
from django.core.paginator import Paginator
from django.db.models import Q, Count
from django.http import JsonResponse, HttpResponse, Http404
from django.shortcuts import render, get_object_or_404, redirect
from django.urls import reverse_lazy, reverse
from django.utils import timezone
from django.views import View
from django.views.generic import ListView, CreateView, UpdateView, DeleteView, DetailView, TemplateView

from .forms import (
  ProjectForm, ProjectImageForm, ReportGenerationForm, ImageFilterForm, BulkImageUploadForm,
  UserPreferencesForm
)
from .forms import RegisterForm
from .models import Project, ProjectImage, GeometricShape, Measurement, Report, UserPreferences
from .utils.auto_detector import detect_and_annotate
from .utils.reports import ReportGenerator, QuickReportGenerator


class ProjectListView(LoginRequiredMixin, ListView):
  model = Project
  template_name = 'core/project_list.html'
  context_object_name = 'projects'
  paginate_by = 12

  def get_queryset(self):
    queryset = Project.objects.filter(created_by=self.request.user, is_active=True)
    queryset = queryset.annotate(
      images_count=Count('images', distinct=True),
      measurement_count=Count('images__shapes', distinct=True),
      shapes_count=Count('images__shapes', distinct=True)
    )

    search = self.request.GET.get('search')
    if search:
      queryset = queryset.filter(
        Q(name__icontains=search) | Q(description__icontains=search)
      )

    return queryset.order_by('-created_at')

  def get_context_data(self, **kwargs):
    context = super().get_context_data(**kwargs)
    context['search_query'] = self.request.GET.get('search', '')
    return context


class ProjectCreateView(LoginRequiredMixin, CreateView):
  model = Project
  form_class = ProjectForm
  template_name = 'core/project_form.html'
  success_url = reverse_lazy('core:project_list')

  def form_valid(self, form):
    form.instance.created_by = self.request.user
    messages.success(self.request, f'Proyecto "{form.instance.name}" creado exitosamente.')
    return super().form_valid(form)


class ProjectDetailView(LoginRequiredMixin, DetailView):
  model = Project
  template_name = 'core/project_detail.html'
  context_object_name = 'project'

  def get_queryset(self):
    return Project.objects.filter(created_by=self.request.user, is_active=True)

  def get_context_data(self, **kwargs):
    context = super().get_context_data(**kwargs)

    filter_form = ImageFilterForm(self.request.GET or None)
    images = self.object.images.all()

    if filter_form.is_valid():
      data = filter_form.cleaned_data

      if data['search']:
        images = images.filter(name__icontains=data['search'])

      if data['calibrated'] == 'yes':
        images = images.filter(is_calibrated=True)
      elif data['calibrated'] == 'no':
        images = images.filter(is_calibrated=False)

      if data['has_measurements'] == 'yes':
        images = images.filter(measurements__isnull=False).distinct()
      elif data['has_measurements'] == 'no':
        images = images.filter(measurements__isnull=True)

    paginator = Paginator(images.order_by('-uploaded_at'), 8)
    page_number = self.request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context.update({
      'filter_form': filter_form,
      'images': page_obj,
      'page_obj': page_obj,
      'upload_form': ProjectImageForm(),
      'bulk_upload_form': BulkImageUploadForm(),
      'report_form': ReportGenerationForm(),
      'project_summary': QuickReportGenerator.generate_measurement_summary(self.object)
    })

    return context


class ProjectUpdateView(LoginRequiredMixin, UpdateView):
  model = Project
  form_class = ProjectForm
  template_name = 'core/project_form.html'

  def get_queryset(self):
    return Project.objects.filter(created_by=self.request.user)

  def get_success_url(self):
    return reverse('core:project_detail', kwargs={'pk': self.object.pk})

  def form_valid(self, form):
    messages.success(self.request, 'Proyecto actualizado exitosamente.')
    return super().form_valid(form)


class ProjectDeleteView(LoginRequiredMixin, DeleteView):
  model = Project
  template_name = 'core/project_confirm_delete.html'
  success_url = reverse_lazy('core:project_list')

  def get_queryset(self):
    return Project.objects.filter(created_by=self.request.user)

  def delete(self, request, *args, **kwargs):
    self.object = self.get_object()
    self.object.is_active = False
    self.object.save()
    messages.success(request, f'Proyecto "{self.object.name}" eliminado exitosamente.')
    return redirect(self.success_url)


class DeleteManualMeasurementView(LoginRequiredMixin, View):
  http_method_names = ['delete']

  def delete(self, request, image_pk, measurement_id):
    try:
      measurement = get_object_or_404(
        Measurement,
        pk=measurement_id,
        image__pk=image_pk,
        image__project__created_by=request.user
      )

      shape_to_delete = measurement.shape

      measurement.delete()

      if shape_to_delete:
        shape_to_delete.delete()

      return JsonResponse({'status': 'success', 'message': 'Medida eliminada correctamente.'})

    except Http404:
      return JsonResponse({'status': 'error', 'message': 'La medida no fue encontrada.'}, status=404)
    except Exception as e:
      logging.getLogger(__name__).error(f"Error al eliminar medida: {e}")
      return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


class ImageDetailView(LoginRequiredMixin, DetailView):
  model = ProjectImage
  template_name = 'core/image_detail.html'
  context_object_name = 'image'

  def get_queryset(self):
    return ProjectImage.objects.filter(project__created_by=self.request.user)

  def get_context_data(self, **kwargs):
    context = super().get_context_data(**kwargs)
    return context


class SaveManualMeasurementView(LoginRequiredMixin, View):
  http_method_names = ['post']

  def post(self, request, image_pk):
    image = get_object_or_404(ProjectImage, pk=image_pk, project__created_by=request.user)

    try:
      data = json.loads(request.body)
      points = data.get('points', [])
      value_pixels = data.get('value_pixels')
      value_real = data.get('value_real')
      unit = data.get('unit', 'cm')

      if not all([points, len(points) >= 2, value_pixels is not None, value_real is not None]):
        return JsonResponse({
          'status': 'error',
          'message': 'Datos de medición incompletos.'
        }, status=400)

      shape = GeometricShape.objects.create(
        image=image,
        name="Línea de Medición Manual",
        shape_type='line',
        points=points,
        detected_label='Manual',
      )

      measurement = Measurement.objects.create(
        image=image,
        shape=shape,
        name=f"Medida Manual #{image.measurements.filter(is_manual=True).count() + 1}",
        measurement_type='manual_distance',
        value_pixels=float(value_pixels),
        value_real=float(value_real),
        unit=unit,
        is_manual=True
      )

      response_data = {
        'status': 'success',
        'message': 'Medición guardada correctamente.',
        'measurement': {
          'id': str(measurement.id),
          'name': measurement.name,
          'value_display': measurement.value_display,
        }
      }
      return JsonResponse(response_data, status=201)

    except Exception as e:
      logging.getLogger(__name__).error(f'Error al guardar medición manual: {str(e)}')
      return JsonResponse({'status': 'error', 'message': f'Error interno del servidor: {str(e)}'}, status=500)


class GetManualMeasurementsView(LoginRequiredMixin, View):
  http_method_names = ['get']

  def get(self, request, image_pk):
    image = get_object_or_404(ProjectImage, pk=image_pk, project__created_by=request.user)

    manual_measurements = Measurement.objects.filter(
      image=image,
      is_manual=True
    ).select_related('shape').order_by('created_at')

    data = []
    for measurement in manual_measurements:
      measurement_data = {
        'id': str(measurement.id),
        'name': measurement.name,
        'value_pixels': measurement.value_pixels,
        'value_real': measurement.value_real,
        'unit': measurement.unit,
      }

      if measurement.shape:
        measurement_data['shape'] = {
          'points': measurement.shape.points
        }

      data.append(measurement_data)

    return JsonResponse({'measurements': data})


class ExportUserDataView(LoginRequiredMixin, View):
  http_method_names = ['get']

  def get(self, request):
    user = request.user

    data = {
      'user': {
        'username': user.username,
        'email': user.email,
        'first_name': user.first_name,
        'last_name': user.last_name,
        'date_joined': user.date_joined.isoformat() if user.date_joined else None,
      },
      'projects': []
    }

    for project in Project.objects.filter(created_by=user).prefetch_related('images__shapes', 'images__measurements'):
      p = {
        'id': str(project.id),
        'name': project.name,
        'description': project.description,
        'created_at': project.created_at.isoformat() if project.created_at else None,
        'images': []
      }

      for image in project.images.all():
        img = {
          'id': str(image.id),
          'name': image.name,
          'image_url': image.image.url if image.image else None,
          'width': image.width,
          'height': image.height,
          'is_calibrated': image.is_calibrated,
          'calibration_data': image.calibration_data,
          'uploaded_at': image.uploaded_at.isoformat() if image.uploaded_at else None,
          'shapes': [],
          'measurements': []
        }

        for shape in image.shapes.all():
          img['shapes'].append({
            'id': str(shape.id),
            'name': shape.name,
            'shape_type': shape.shape_type,
            'points': shape.points,
            'properties': shape.properties,
            'created_at': shape.created_at.isoformat() if shape.created_at else None,
          })

        for meas in image.measurements.all():
          img['measurements'].append({
            'id': str(meas.id),
            'name': meas.name,
            'type': meas.measurement_type,
            'value_pixels': meas.value_pixels,
            'value_real': meas.value_real,
            'unit': meas.unit,
            'created_at': meas.created_at.isoformat() if meas.created_at else None,
          })

        p['images'].append(img)

      data['projects'].append(p)

    payload = json.dumps(data, ensure_ascii=False, indent=2)
    filename = f'user_data_{user.username}.json'

    response = HttpResponse(payload, content_type='application/json; charset=utf-8')
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response


class DeleteAccountView(LoginRequiredMixin, View):
  http_method_names = ['post']

  def post(self, request):
    user = request.user
    try:
      data = json.loads(request.body or '{}')
    except Exception:
      return JsonResponse({'success': False, 'error': 'invalid_request'}, status=400)

    password = data.get('password')
    if not password:
      return JsonResponse({'success': False, 'error': 'password_required'}, status=400)

    if not user.check_password(password):
      return JsonResponse({'success': False, 'error': 'invalid_password'}, status=403)

    try:
      logout(request)
      username = user.username
      user.delete()
      return JsonResponse({'success': True, 'redirect_url': '/'})
    except Exception as e:
      logging.getLogger(__name__).exception('Failed to delete account for user %s: %s', user.pk, e)
      return JsonResponse({'success': False, 'error': str(e)}, status=500)


class UploadImageView(LoginRequiredMixin, View):
  http_method_names = ['post']

  def post(self, request, project_pk):
    project = get_object_or_404(Project, pk=project_pk, created_by=request.user)
    form = ProjectImageForm(request.POST, request.FILES)

    if form.is_valid():
      image = form.save(commit=False)
      image.project = project
      image.save()

      messages.success(request, f'Imagen "{image.name}" subida exitosamente.')
      return redirect('core:project_detail', pk=project.pk)
    else:
      messages.error(request, 'Error al subir la imagen. Verifique los datos.')
      return redirect('core:project_detail', pk=project.pk)


class BulkUploadImagesView(LoginRequiredMixin, View):
  http_method_names = ['post']

  def post(self, request, project_pk):
    project = get_object_or_404(Project, pk=project_pk, created_by=request.user)
    form = BulkImageUploadForm(request.POST, request.FILES)

    if form.is_valid():
      images = request.FILES.getlist('images')
      prefix = form.cleaned_data.get('name_prefix', '')
      uploaded_count = 0

      for image_file in images:
        base_name = image_file.name.rsplit('.', 1)[0]
        name = f"{prefix}{base_name}" if prefix else base_name

        try:
          project_image = ProjectImage(
            project=project,
            name=name,
            image=image_file
          )
          project_image.save()
          uploaded_count += 1
        except Exception as e:
          messages.warning(request, f'Error al subir {image_file.name}: {str(e)}')

      if uploaded_count > 0:
        messages.success(request, f'{uploaded_count} imágenes subidas exitosamente.')

      return redirect('core:project_detail', pk=project.pk)
    else:
      messages.error(request, 'Error en la subida múltiple. Verifique los archivos.')
      return redirect('core:project_detail', pk=project.pk)


class GenerateReportView(LoginRequiredMixin, View):
  http_method_names = ['post']

  def post(self, request, project_pk):
    project = get_object_or_404(Project, pk=project_pk, created_by=request.user)
    form = ReportGenerationForm(request.POST)

    if form.is_valid():
      data = form.cleaned_data
      generator = ReportGenerator(project)

      try:
        include_images = bool(data.get('include_images'))
        include_measurements = bool(data.get('include_measurements'))
        include_shapes = bool(data.get('include_shapes'))

        if data['format'] == 'pdf':
          filepath = generator.generate_pdf_report(
            include_images=include_images,
            include_measurements=include_measurements,
            include_shapes=include_shapes
          )
        elif data['format'] == 'excel':
          filepath = generator.generate_excel_report(
            include_images=include_images,
            include_measurements=include_measurements,
            include_shapes=include_shapes
          )
        elif data['format'] == 'json':
          filepath = generator.generate_json_report(
            include_images=include_images,
            include_measurements=include_measurements,
            include_shapes=include_shapes
          )

        report = Report.objects.create(
          project=project,
          name=f"Reporte {data['format'].upper()} - {timezone.now().strftime('%Y-%m-%d %H:%M')}",
          format=data['format'],
          file_path=filepath,
          parameters=data
        )

        messages.success(request, 'Reporte generado exitosamente.')
        return redirect('core:download_report', report_pk=report.pk)

      except Exception as e:
        messages.error(request, f'Error al generar el reporte: {str(e)}')

    return redirect('core:project_detail', pk=project.pk)


class DownloadReportView(LoginRequiredMixin, View):
  http_method_names = ['get']

  def get(self, request, report_pk):
    report = get_object_or_404(Report, pk=report_pk, project__created_by=request.user)

    if not report.file_exists:
      raise Http404("El archivo del reporte no existe")

    with open(report.file_path, 'rb') as f:
      response = HttpResponse(f.read())

      if report.format == 'pdf':
        response['Content-Type'] = 'application/pdf'
      elif report.format == 'excel':
        response['Content-Type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
      elif report.format == 'json':
        response['Content-Type'] = 'application/json'

      filename = os.path.basename(report.file_path)
      response['Content-Disposition'] = f'attachment; filename="{filename}"'

      return response


class ArucoCheckView(LoginRequiredMixin, View):
  http_method_names = ['get']

  def get(self, request, image_pk):
    image = get_object_or_404(ProjectImage, pk=image_pk, project__created_by=request.user)
    image_path = image.image.path if image.image else None
    if not image_path or not os.path.exists(image_path):
      return JsonResponse({'error': 'Image file not found', 'aruco_found': False}, status=404)

    try:
      img = cv2.imread(image_path)
      if img is None:
        return JsonResponse({'error': 'Unable to read image', 'aruco_found': False}, status=500)

      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        corners_list, ids, _ = detector.detectMarkers(gray)
      except Exception:
        corners_list, ids = None, None

      if ids is None or len(ids) == 0:
        return JsonResponse({'aruco_found': False, 'ids': []})

      c = corners_list[0].reshape((4, 2))
      lado1 = np.linalg.norm(c[0] - c[1])
      lado2 = np.linalg.norm(c[1] - c[2])
      lado3 = np.linalg.norm(c[2] - c[3])
      lado4 = np.linalg.norm(c[3] - c[0])
      avg_pixels = float(np.mean([lado1, lado2, lado3, lado4]))

      return JsonResponse({'aruco_found': True, 'side_px': avg_pixels, 'ids': ids.flatten().tolist()})
    except Exception as e:
      logging.exception('Error in aruco_check')
      return JsonResponse({'aruco_found': False, 'error': str(e)}, status=500)


class ImageDataAPIView(LoginRequiredMixin, View):
  http_method_names = ['get']

  def get(self, request, image_pk):
    image = get_object_or_404(ProjectImage, pk=image_pk, project__created_by=request.user)

    data = {
      'image': {
        'id': str(image.id),
        'name': image.name,
        'width': image.width,
        'height': image.height,
        'is_calibrated': image.is_calibrated,
        'calibration_unit': image.calibration_unit,
        'scale_factor': image.scale_factor,
        'annotated_url': image.calibration_data.get('annotated_url') if image.calibration_data else None,
        'url': image.image.url
      },
      'shapes': [],
      'measurements': []
    }

    for shape in image.shapes.all():
      data['shapes'].append({
        'id': str(shape.id),
        'name': shape.name,
        'type': shape.shape_type,
        'detected_label': getattr(shape, 'detected_label', None),
        'points': shape.points,
        'properties': shape.properties,
        'center_px': shape.properties.get('raw', {}).get('center_px') if shape.properties else None,
        'color': shape.color,
        'annotated_url': data['image'].get('annotated_url'),
        'triangle_type': shape.properties.get('triangle_type') if shape.shape_type == 'triangle' else None,
        'sides_cm': shape.properties.get('sides_cm') if shape.shape_type == 'triangle' else None
      })

    for measurement in image.measurements.all():
      measurement_data = {
        'id': str(measurement.id),
        'name': measurement.name,
        'type': measurement.measurement_type,
        'shape_id': str(measurement.shape.id) if measurement.shape else None,
        'value_pixels': measurement.value_pixels,
        'value_real': measurement.value_real,
        'unit': measurement.unit,
        'is_manual': measurement.measurement_type == 'manual_distance',
        'value_display': measurement.value_display
      }
      data['measurements'].append(measurement_data)

    return JsonResponse(data)


class ProcessImageDetectionView(LoginRequiredMixin, View):
  http_method_names = ['post']

  def post(self, request, image_pk):
    image = get_object_or_404(ProjectImage, pk=image_pk, project__created_by=request.user)
    logger = logging.getLogger(__name__)
    created_shapes_summary = []
    created_measurements_summary = []

    # 1. Limpiar detecciones anteriores (código original)
    try:
      image.shapes.filter(is_auto_detected=True).delete()
      logger.info(f"Limpieza de detecciones anteriores para la imagen {image.pk} completada.")
    except Exception as e:
      logger.exception(f'Error al limpiar detecciones para la imagen {image.pk}: {e}')

    # 2. Procesar la imagen usando la ruta local del sistema de archivos (código original)
    try:
      detection = detect_and_annotate(image.image.path, aruco_real_size_cm=5.0)
    except Exception as e:
      logger.exception(f"Error en 'detect_and_annotate' para la imagen {image.pk}: {e}")
      return JsonResponse({'status': 'error', 'message': f"Error al procesar la imagen: {e}"}, status=500)

    # --- INICIO DEL BLOQUE PARA SUBIR A S3 MANTENIENDO EL LOCAL ---
    annotated_path = detection.get('annotated_path')
    annotated_url = detection.get('annotated_url')

    if annotated_path and os.path.exists(annotated_path):
        logger.info(f"Imagen procesada guardada localmente en: {annotated_path}")
        try:
            # Crear un cliente de S3
            s3_client = boto3.client(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_S3_REGION_NAME
            )

            # Definir la ruta dentro del bucket de S3
            file_name = os.path.basename(annotated_path)
            s3_key = f"media/processed/{file_name}"

            # Subir el archivo
            s3_client.upload_file(
                annotated_path,
                settings.AWS_STORAGE_BUCKET_NAME,
                s3_key,
                ExtraArgs={'ContentType': 'image/jpeg'}
            )
            
            # Construir la URL pública de S3 y sobreescribir la URL local
            annotated_url = f"https://{settings.AWS_S3_CUSTOM_DOMAIN}/{s3_key}"
            detection['annotated_url'] = annotated_url
            
            logger.info(f"Éxito: Imagen procesada también subida a S3 en: {annotated_url}")

        except Exception as upload_error:
            logger.error(f"FALLO al subir la imagen procesada a S3: {upload_error}. Se usará la URL local.")
            # Si falla la subida, no es crítico. La URL seguirá siendo la local.
    # --- FIN DEL BLOQUE PARA SUBIR A S3 ---

    results = detection.get('results', [])
    pixels_per_cm = detection.get('pixels_per_cm')

    # 3. Guardar datos de calibración con la URL final (S3 o local)
    if pixels_per_cm:
      try:
        image.is_calibrated = True
        image.calibration_data.update({
          'reference': 'aruco', 'pixels_per_cm': pixels_per_cm, 'unit': 'cm',
          'annotated_url': detection.get('annotated_url')
        })
        image.save()
      except Exception as e:
        logger.exception('Falló al guardar datos de calibración para la imagen %s: %s', image.pk, e)
    
    # 4. Crear formas geométricas y mediciones (código original)
    for obj in results:
      try:
        idx = obj.get('index')
        detected_label = str(obj.get('shape') or 'Objeto')
        shape_type_field = 'polygon'
        if 'círculo' in detected_label.lower():
          shape_type_field = 'circle'
        elif 'triángulo' in detected_label.lower():
          shape_type_field = 'triangle'
        elif 'rectángulo' in detected_label.lower():
          shape_type_field = 'rectangle'

        shape_name = f"{detected_label.capitalize()} ({idx})"
        points = [{'x': p[0], 'y': p[1]} for p in obj.get('box_px', [])]

        gs = GeometricShape.objects.create(
          image=image, name=shape_name, shape_type=shape_type_field,
          points=points, detected_label=detected_label, properties=obj,
          is_auto_detected=True
        )
        created_shapes_summary.append({'id': str(gs.id), 'name': gs.name})

        measurement_defs = {'area': ('Área', 'area_px', 'area_cm2', 'cm²'),
                            'perimeter': ('Perímetro', 'perimeter_px', 'perimeter_cm', 'cm'),
                            'width': ('Ancho', 'w_px', 'w_cm', 'cm'), 'height': ('Alto', 'h_px', 'h_cm', 'cm'),
                            'radius': ('Radio', 'radius_px', 'radius_cm', 'cm'),
                            'diameter': ('Diámetro', 'diameter_px', 'diameter_cm', 'cm')}
        for key, (name, px, real, unit) in measurement_defs.items():
          if px in obj and real in obj:
            meas = Measurement.objects.create(image=image, shape=gs, name=f"{name} {gs.name}", measurement_type=key,
                                              value_pixels=obj[px], value_real=obj[real], unit=unit)
            created_measurements_summary.append({'id': str(meas.id), 'name': meas.name})

      except Exception as e:
        logger.exception(f"Falló al guardar el objeto detectado {obj.get('index', 'N/A')}: {e}")
        continue

    # 5. Envío de notificación por correo (código original)
    email_sent = False
    try:
      preferences = UserPreferences.get_or_create_for_user(request.user)
      if preferences.email_notifications and request.user.email and results:
        subject = f"Detección completada: {len(results)} objeto(s) en '{image.name}'"

        body_lines = [
          f"Estimado/a {request.user.first_name or request.user.username},",
          f"\nLa detección automática sobre la imagen '{image.name}' ha finalizado. Se han detectado {len(results)} objeto(s).",
          "\nResumen de las mediciones:"
        ]
        for obj in results:
          line = f"- {obj.get('shape', 'Objeto')} ({obj.get('index')}): Área={obj.get('area_cm2', 0):.2f} cm², Perímetro={obj.get('perimeter_cm', 0):.2f} cm"
          body_lines.append(line)

        body_lines.append(f"\nPuede ver los resultados detallados en el siguiente enlace:")
        image_url = request.build_absolute_uri(reverse('core:image_detail', kwargs={'pk': image.pk}))
        body_lines.append(image_url)
        body_lines.append("\nAtentamente,\nEl equipo de Medición Geométrica")

        email = EmailMessage(
          subject, '\n'.join(body_lines),
          settings.DEFAULT_FROM_EMAIL, [request.user.email]
        )

        try:
          generator = ReportGenerator(image.project)
          pdf_path = generator.generate_pdf_for_image(image)
          excel_path = generator.generate_excel_for_image(image)
          if pdf_path and os.path.exists(pdf_path):
            email.attach_file(pdf_path)
          if excel_path and os.path.exists(excel_path):
            email.attach_file(excel_path)
        except Exception as report_error:
          logger.error(f"No se pudieron adjuntar los reportes al email: {report_error}")

        sent_count = email.send(fail_silently=False)
        email_sent = sent_count > 0

    except Exception as email_error:
      logger.exception(
        f"Falló el envío de la notificación por email para el usuario {request.user.username}: {email_error}")

    # 6. Respuesta final (código original)
    return JsonResponse({
      'status': 'success',
      'message': f'Detección completada. {len(results)} objetos encontrados.',
      'results': results,
      'shapes': created_shapes_summary,
      'measurements': created_measurements_summary,
      'annotated_url': detection.get('annotated_url'),
      'email_sent': email_sent,
      'calibration_unit': image.calibration_unit,
      'scale_factor': pixels_per_cm
    })


class PreviewImagePDFView(LoginRequiredMixin, View):
  http_method_names = ['get']

  def get(self, request, image_pk):
    image = get_object_or_404(ProjectImage, pk=image_pk, project__created_by=request.user)
    project = image.project
    generator = ReportGenerator(project)

    try:
      filepath = generator.generate_pdf_for_image(image, include_measurements=True)
      with open(filepath, 'rb') as f:
        response = HttpResponse(f.read(), content_type='application/pdf')
        filename = os.path.basename(filepath)
        response['Content-Disposition'] = f'inline; filename="{filename}"'
        return response
    except Exception as e:
      messages.error(request, f'Error al generar PDF: {str(e)}')
      return redirect('core:image_detail', pk=image.pk)


class DownloadImageExcelView(LoginRequiredMixin, View):
  http_method_names = ['get']

  def get(self, request, image_pk):
    image = get_object_or_404(ProjectImage, pk=image_pk, project__created_by=request.user)
    project = image.project
    generator = ReportGenerator(project)

    try:
      filepath = generator.generate_excel_for_image(image)
      with open(filepath, 'rb') as f:
        response = HttpResponse(f.read(),
                                content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        filename = os.path.basename(filepath)
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response
    except Exception as e:
      messages.error(request, f'Error al generar Excel: {str(e)}')
      return redirect('core:image_detail', pk=image.pk)


class DashboardView(LoginRequiredMixin, ListView):
  template_name = 'core/dashboard.html'
  context_object_name = 'recent_projects'
  paginate_by = 6

  def get_queryset(self):
    return Project.objects.filter(
      created_by=self.request.user,
      is_active=True
    ).annotate(
      image_count=Count('images'),
      measurement_count=Count('images__shapes')
    ).order_by('-updated_at')

  def get_context_data(self, **kwargs):
    context = super().get_context_data(**kwargs)

    user_projects = Project.objects.filter(created_by=self.request.user, is_active=True)

    user_images = ProjectImage.objects.filter(project__in=user_projects).prefetch_related('shapes', 'measurements')
    total_measurements = sum(
      getattr(img, 'logical_measurements_count', img.measurements.count())
      for img in user_images
    )

    context['stats'] = {
      'total_projects': user_projects.count(),
      'total_images': user_images.count(),
      'total_measurements': total_measurements,
      'calibrated_images': user_images.filter(is_calibrated=True).count()
    }

    return context


class GalleryView(LoginRequiredMixin, ListView):
  template_name = 'core/gallery.html'
  context_object_name = 'images'
  paginate_by = 20

  def get_queryset(self):
    return ProjectImage.objects.filter(
      project__created_by=self.request.user
    ).select_related('project').prefetch_related('measurements', 'shapes').order_by('-uploaded_at')

  def get_context_data(self, **kwargs):
    context = super().get_context_data(**kwargs)

    project_filter = self.request.GET.get('project')
    calibrated_filter = self.request.GET.get('calibrated')
    search_query = self.request.GET.get('search')

    queryset = self.get_queryset()

    if project_filter:
      queryset = queryset.filter(project_id=project_filter)

    if calibrated_filter == 'true':
      queryset = queryset.filter(is_calibrated=True)
    elif calibrated_filter == 'false':
      queryset = queryset.filter(is_calibrated=False)

    if search_query:
      queryset = queryset.filter(name__icontains=search_query)

    context['images'] = queryset
    context['projects'] = Project.objects.filter(created_by=self.request.user, is_active=True)
    context['total_images'] = queryset.count()
    context['calibrated_count'] = queryset.filter(is_calibrated=True).count()

    return context


class StatisticsView(LoginRequiredMixin, TemplateView):
  template_name = 'core/statistics.html'

  def get_context_data(self, **kwargs):
    context = super().get_context_data(**kwargs)

    user_projects = Project.objects.filter(created_by=self.request.user, is_active=True)
    user_images = ProjectImage.objects.filter(project__in=user_projects).prefetch_related('shapes', 'measurements')
    user_measurements = Measurement.objects.filter(image__project__in=user_projects)

    total_images = user_images.count()
    calibrated_images = user_images.filter(is_calibrated=True).count()

    total_measurements = sum(
      getattr(img, 'logical_measurements_count', img.measurements.count())
      for img in user_images
    )

    context['stats'] = {
      'total_projects': user_projects.count(),
      'total_images': total_images,
      'total_measurements': total_measurements,
      'total_shapes': GeometricShape.objects.filter(image__project__in=user_projects).count(),
      'calibrated_images': calibrated_images,
      'active_projects': user_projects.filter(is_active=True).count(),
      'calibrated_percentage': round((calibrated_images / total_images * 100), 0) if total_images > 0 else 0,
    }

    from datetime import datetime, timedelta
    thirty_days_ago = datetime.now() - timedelta(days=30)

    recent_images = user_images.filter(uploaded_at__gte=thirty_days_ago)
    new_measurements = sum(
      getattr(img, 'logical_measurements_count', img.measurements.count())
      for img in recent_images
    )

    context['recent_activity'] = {
      'new_projects': user_projects.filter(created_at__gte=thirty_days_ago).count(),
      'new_images': recent_images.count(),
      'new_measurements': new_measurements,
    }

    project_stats = []
    for project in user_projects.order_by('-created_at')[:10]:
      imgs = project.images.all().prefetch_related('shapes', 'measurements')
      image_count = imgs.count()
      measurement_count = sum(getattr(img, 'logical_measurements_count', img.measurements.count()) for img in imgs)
      shape_count = GeometricShape.objects.filter(image__project=project).count()

      has_calibrated = imgs.filter(is_calibrated=True).exists()
      display_active = True if (image_count > 0 or has_calibrated) else False

      project_stats.append({
        'name': project.name,
        'description': project.description,
        'image_count': image_count,
        'measurement_count': measurement_count,
        'shape_count': shape_count,
        'is_active': display_active,
        'created_at': project.created_at,
        'pk': project.pk,
      })

    context['project_stats'] = project_stats

    measurement_types = user_measurements.values('measurement_type').annotate(
      count=Count('id')
    ).order_by('-count')
    context['measurement_types'] = list(measurement_types)

    monthly_data = []
    for i in range(12):
      month_start = datetime.now() - timedelta(days=30 * i)
      month_end = month_start + timedelta(days=30)

      images_in_month = user_images.filter(uploaded_at__gte=month_start, uploaded_at__lt=month_end)
      measurements_in_month = sum(
        getattr(img, 'logical_measurements_count', img.measurements.count()) for img in images_in_month)

      monthly_data.append({
        'month': month_start.strftime('%b'),
        'projects': user_projects.filter(created_at__gte=month_start, created_at__lt=month_end).count(),
        'images': images_in_month.count(),
        'measurements': measurements_in_month,
      })

    context['monthly_data'] = list(reversed(monthly_data))

    return context


class UserProfileView(LoginRequiredMixin, TemplateView):
  template_name = 'core/user_profile.html'

  def get_context_data(self, **kwargs):
    context = super().get_context_data(**kwargs)

    user = self.request.user
    user_projects = Project.objects.filter(created_by=user, is_active=True)
    user_images = ProjectImage.objects.filter(project__in=user_projects).prefetch_related('shapes', 'measurements')

    total_measurements = sum(
      getattr(img, 'logical_measurements_count', img.measurements.count())
      for img in user_images
    )

    context['user_stats'] = {
      'total_projects': user_projects.count(),
      'total_images': user_images.count(),
      'total_measurements': total_measurements,
      'calibrated_images': user_images.filter(is_calibrated=True).count(),
      'member_since': user.date_joined,
      'last_login': user.last_login,
    }

    context['recent_projects'] = user_projects.order_by('-created_at')[:5]

    context['recent_images'] = user_images.order_by('-uploaded_at')[:10]

    most_active_project = None
    best_score = (0, 0)
    for p in user_projects:
      imgs = p.images.all().prefetch_related('shapes', 'measurements')
      img_count = imgs.count()
      meas_count = sum(getattr(img, 'logical_measurements_count', img.measurements.count()) for img in imgs)
      if img_count > best_score[0] or (img_count == best_score[0] and meas_count > best_score[1]):
        most_active_project = p
        best_score = (img_count, meas_count)

    context['most_active_project'] = most_active_project

    return context


class UserSettingsView(LoginRequiredMixin, TemplateView):
  template_name = 'core/user_settings.html'

  def get_context_data(self, **kwargs):
    context = super().get_context_data(**kwargs)

    preferences = UserPreferences.get_or_create_for_user(self.request.user)

    context['account_info'] = {
      'username': self.request.user.username,
      'email': self.request.user.email,
      'first_name': self.request.user.first_name,
      'last_name': self.request.user.last_name,
      'date_joined': self.request.user.date_joined,
      'last_login': self.request.user.last_login,
    }

    context['preferences_form'] = UserPreferencesForm(instance=preferences)

    return context

  def post(self, request, *args, **kwargs):
    preferences = UserPreferences.get_or_create_for_user(request.user)
    form = UserPreferencesForm(request.POST, instance=preferences)

    if form.is_valid():
      form.save()
      messages.success(request, 'Configuración guardada exitosamente.')
      return redirect('core:user_settings')
    else:
      messages.error(request, 'Error al guardar la configuración. Verifica los datos.')

    context = self.get_context_data()
    context['preferences_form'] = form
    return self.render_to_response(context)


class UpdateThemeView(LoginRequiredMixin, View):
  http_method_names = ['post']
  csrf_exempt = True

  def post(self, request):
    try:
      data = json.loads(request.body)
      theme = data.get('theme')

      if theme not in ['light', 'dark']:
        return JsonResponse({'success': False, 'error': 'Invalid theme'})

      preferences = UserPreferences.get_or_create_for_user(request.user)
      preferences.theme = theme
      preferences.save()

      return JsonResponse({'success': True})
    except Exception as e:
      return JsonResponse({'success': False, 'error': str(e)})


class HelpView(TemplateView):
  template_name = 'core/help.html'

  def get_context_data(self, **kwargs):
    context = super().get_context_data(**kwargs)

    context['help_sections'] = [
      {
        'id': 'getting-started',
        'title': 'Comenzando',
        'icon': 'fas fa-play-circle',
        'sections': [
          {
            'title': 'Crear un proyecto',
            'content': 'Un proyecto agrupa imágenes y sus mediciones. Crea uno por cada conjunto de fotos relacionadas (ej. un equipo, una muestra, una sesión).',
            'steps': [
              'Haz clic en "Nuevo Proyecto" desde el menú o el dashboard',
              'Rellena el nombre y una descripción breve',
              'Confirma para crear el proyecto y serás llevado a su detalle'
            ],
            'tips': [
              'Usa nombres descriptivos para identificar fácilmente el contexto de las mediciones',
            ]
          },
          {
            'title': 'Subir imágenes',
            'content': 'Puedes subir una o varias imágenes al proyecto. Cada imagen se procesa y queda lista para calibración y detección.',
            'steps': [
              'En el detalle del proyecto usa "Subir Imagen" o "Subida Múltiple"',
              'Selecciona los archivos (JPEG, PNG, BMP, TIFF). Evita archivos muy grandes para velocidad',
              'Espera a que las miniaturas aparezcan en la galería del proyecto'
            ],
            'tips': [
              'Si vas a procesar muchas imágenes usa la opción de subida múltiple y espera a que termine el proceso',
            ]
          }
        ]
      },
      {
        'id': 'calibration',
        'title': 'Calibración',
        'icon': 'fas fa-ruler',
        'sections': [
          {
            'title': 'Por qué calibrar',
            'content': 'La calibración convierte píxeles en unidades reales (cm, mm). Sin calibración los valores solo estarán en píxeles.',
            'steps': [
              'Abre la imagen y localiza un objeto de tamaño conocido (regla, moneda, marca ArUco)',
              'Usa la herramienta de calibración para dibujar una línea sobre ese objeto',
              'Introduce la medida real y la unidad, y guarda la calibración'
            ],
            'tips': [
              'Asegúrate de que la referencia esté en el mismo plano que lo que vas a medir',
              'Una línea más larga suele reducir el error relativo'
            ]
          },
          {
            'title': 'Calibración automática (ArUco)',
            'content': 'Cuando el sistema detecta un marcador ArUco, puede calibrar la imagen automáticamente y marcarla como "Calibrada".',
            'steps': [
              'Sube una imagen que contenga un marcador ArUco de referencia',
              'Haz clic en "Realizar Medición" (o en la detección automática)',
              'Si se detecta el marcador, la calibración se aplicará automáticamente'
            ]
          }
        ]
      },
      {
        'id': 'measurements',
        'title': 'Mediciones',
        'icon': 'fas fa-ruler-combined',
        'sections': [
          {
            'title': 'Herramientas de medición',
            'content': 'Tras calibrar una imagen puedes usar las herramientas para medir distancias, perímetros y áreas directamente sobre la imagen.',
            'steps': [
              'Abre la imagen desde el proyecto o la galería',
              'Selecciona la herramienta (línea, polígono, círculo) según lo que quieras medir',
              'Dibuja sobre la imagen y guarda la medición con un nombre descriptivo'
            ],
            'types': [
              'Distancia: entre dos puntos',
              'Área: forma cerrada (polígono/círculo)',
              'Perímetro: contorno de una forma cerrada'
            ],
            'tips': [
              'Nombra las mediciones para identificar resultados en reportes',
            ]
          },
          {
            'title': 'Cómo se contabilizan las mediciones',
            'content': 'Cada forma detectada genera varios valores (ancho, alto, área). Por eso en algunas tablas verás más filas de "mediciones" que de "formas". En los resúmenes por proyecto el sistema muestra el número de formas detectadas cuando se busca representar "mediciones lógicas" (una por objeto).'
          }
        ]
      },
      {
        'id': 'shape-detection',
        'title': 'Detección automática',
        'icon': 'fas fa-shapes',
        'sections': [
          {
            'title': 'Realizar Medición (flujo automático)',
            'content': 'Un único botón "Realizar Medición" envía la imagen al servidor para detectar formas automáticamente, guarda las formas (GeometricShape) y las mediciones derivadas en la base de datos, y muestra los resultados inmediatamente en la interfaz.',
            'steps': [
              'Abre la imagen que quieras procesar',
              'Haz clic en "Realizar Medición"',
              'Espera a que finalice el proceso (la interfaz mostrará una notificación)',
              'Revisa las formas detectadas en la lista y en el lienzo; ahora estarán persistidas'
            ],
            'tips': [
              'Si ya existe una detección automática anterior, el proceso remueve las formas/mediciones anteriores con nombre similar y crea las nuevas (idempotente para ese flujo)',
              'Si no ves formas después de la detección, revisa el log o el mensaje de la interfaz para errores'
            ],
            'parameters': [
              'Sensibilidad de detección: controla cuánto debe coincidir una forma para considerarse válida',
              'Área mínima: descarta formas muy pequeñas'
            ]
          }
        ]
      },
      {
        'id': 'reports',
        'title': 'Reportes',
        'icon': 'fas fa-file-alt',
        'sections': [
          {
            'title': 'Generar reportes por proyecto o imagen',
            'content': 'Puedes generar reportes que incluyan imágenes y mediciones; los reportes se guardan como archivos que puedes descargar.',
            'steps': [
              'Desde el detalle del proyecto o la vista de imagen, abre el generador de reportes',
              'Elige formato (PDF, Excel, JSON) y opciones (incluir imágenes, incluir todas las mediciones, filtros)',
              'Genera y descarga el archivo; el sistema registra el reporte en la lista de reportes del proyecto'
            ]
          },
          {
            'title': 'Formatos y usos',
            'content': 'Cada formato tiene un uso: PDF para presentaciones, Excel para análisis y JSON para integración con otros sistemas.'
          }
        ]
      },
      {
        'id': 'troubleshooting',
        'title': 'Solución de Problemas',
        'icon': 'fas fa-tools',
        'sections': [
          {
            'title': 'Problemas comunes y soluciones',
            'content': 'Lista de errores frecuentes y acciones recomendadas:',
            'problems': [
              'La imagen no carga: comprueba la ruta del archivo y el permiso de lectura; revisa el tamaño y formato',
              'La calibración no funciona: verifica que la referencia esté visible y sea del tamaño correcto',
              'No se detectan formas: prueba aumentando la calidad/contraste de la imagen o ajusta los parámetros de detección',
              'Resultados inesperados en mediciones: revisa si la imagen está deformada o fuera de foco'
            ]
          },
          {
            'title': 'Requisitos y recomendaciones',
            'content': 'Para un funcionamiento óptimo recomendamos:',
            'requirements': [
              'Usar un navegador moderno actualizado (Chrome/Edge/Firefox)',
              'Subir imágenes con buena iluminación y contraste',
              'Incluir un objeto de referencia o marcador ArUco para calibraciones automáticas',
              'Evitar compresión excesiva (JPEG con mucha pérdida)'
            ]
          }
        ]
      }
    ]

    try:
      from django.conf import settings
      import urllib.parse

      support_email = getattr(settings, 'EMAIL_HOST_USER', None) or getattr(settings, 'DEFAULT_FROM_EMAIL')

      user_first = (self.request.user.first_name or '').strip()
      user_last = (self.request.user.last_name or '').strip()
      user_fullname = ' '.join(p for p in (user_first, user_last) if p) or self.request.user.username
      user_email = self.request.user.email or ''

      subject = f"Soporte - MediciónGeo"

      body = (
        f"Estimado equipo de soporte,\n\n"
        f"Mi nombre es {user_fullname}.\n"
        "( Por favor describa aquí su consulta: )\n"
      )

      gmail_query = {
        'view': 'cm',
        'fs': '1',
        'to': support_email,
        'su': subject,
        'body': body,
      }
      gmail_compose_url = 'https://mail.google.com/mail/?' + urllib.parse.urlencode(gmail_query)
      context['gmail_compose_url'] = gmail_compose_url
      context['support_email'] = support_email
    except Exception:
      context['gmail_compose_url'] = f"mailto:soporte@mediciongeo.com"
      context['support_email'] = 'soporte@mediciongeo.com'

    return context


class RegisterView(View):
  template_name = 'registration/register.html'

  def get(self, request, *args, **kwargs):
    form = RegisterForm()
    return render(request, self.template_name, {'form': form})

  def post(self, request, *args, **kwargs):
    form = RegisterForm(request.POST)
    if form.is_valid():
      user = form.save()
      messages.success(request, 'Registro exitoso. Por favor, inicia sesión con tus credenciales.')
      return redirect('login')
    else:
      return render(request, self.template_name, {'form': form})
