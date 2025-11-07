import os
import uuid

from django.contrib.auth.models import User
from django.db import models


def upload_to_projects(instance, filename):
  ext = filename.split('.')[-1]
  filename = f"{uuid.uuid4().hex}.{ext}"
  return f"projects/{instance.project.id}/{filename}"


class Project(models.Model):
  id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
  name = models.CharField(max_length=200, verbose_name="Nombre del proyecto")
  description = models.TextField(blank=True, verbose_name="Descripción")
  created_by = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name="Creado por")
  created_at = models.DateTimeField(auto_now_add=True, verbose_name="Fecha de creación")
  updated_at = models.DateTimeField(auto_now=True, verbose_name="Última actualización")
  is_active = models.BooleanField(default=True, verbose_name="Activo")

  class Meta:
    ordering = ['-created_at']
    verbose_name = "Proyecto"
    verbose_name_plural = "Proyectos"

  def __str__(self):
    return self.name

  @property
  def total_images(self):
    return self.images.count()

  @property
  def total_measurements(self):
    return sum(img.shapes.count() for img in self.images.all())


class ProjectImage(models.Model):
  id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
  project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='images')
  name = models.CharField(max_length=200, verbose_name="Nombre de la imagen")
  image = models.ImageField(upload_to=upload_to_projects, verbose_name="Imagen")
  width = models.PositiveIntegerField(verbose_name="Ancho en píxeles")
  height = models.PositiveIntegerField(verbose_name="Alto en píxeles")
  uploaded_at = models.DateTimeField(auto_now_add=True, verbose_name="Fecha de subida")

  is_calibrated = models.BooleanField(default=False, verbose_name="Está calibrada")
  calibration_data = models.JSONField(
    default=dict,
    help_text="Datos de calibración: puntos de referencia y escala"
  )

  class Meta:
    ordering = ['-uploaded_at']
    verbose_name = "Imagen del proyecto"
    verbose_name_plural = "Imágenes del proyecto"

  def __str__(self):
    return f"{self.project.name} - {self.name}"

  def save(self, *args, **kwargs):
    if self.image and not self.width:
      from PIL import Image
      img = Image.open(self.image)
      self.width, self.height = img.size
    super().save(*args, **kwargs)

  @property
  def scale_factor(self):
    if not self.is_calibrated or not self.calibration_data:
      return None

    cal_data = self.calibration_data
    if 'reference_points' not in cal_data or 'real_distance' not in cal_data:
      return None

    p1 = cal_data['reference_points'][0]
    p2 = cal_data['reference_points'][1]
    pixel_distance = ((p2['x'] - p1['x']) ** 2 + (p2['y'] - p1['y']) ** 2) ** 0.5

    return pixel_distance / cal_data['real_distance']

  @property
  def logical_measurements_count(self):
    return self.shapes.count()

  @property
  def calibration_unit(self):
    if not self.is_calibrated or not self.calibration_data:
      return 'px'
    return self.calibration_data.get('unit', 'px')


class GeometricShape(models.Model):
  SHAPE_TYPES = [
    ('point', 'Punto'),
    ('line', 'Línea'),
    ('rectangle', 'Rectángulo'),
    ('circle', 'Círculo'),
    ('triangle', 'Triángulo'),
    ('polygon', 'Polígono'),
  ]

  TRIANGLE_TYPES = [
    ('equilatero', 'Equilátero'),
    ('isosceles', 'Isósceles'),
    ('escaleno', 'Escaleno'),
  ]

  id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
  image = models.ForeignKey(ProjectImage, on_delete=models.CASCADE, related_name='shapes')
  name = models.CharField(max_length=100, verbose_name="Nombre de la forma")
  shape_type = models.CharField(max_length=20, choices=SHAPE_TYPES, verbose_name="Tipo de forma")
  detected_label = models.CharField(max_length=100, null=True, blank=True, verbose_name="Etiqueta detectada")
  points = models.JSONField(help_text="Puntos que definen la forma geométrica")
  properties = models.JSONField(
    default=dict,
    help_text="Propiedades calculadas: área, perímetro, etc."
  )
  created_at = models.DateTimeField(auto_now_add=True, verbose_name="Fecha de creación")
  color = models.CharField(max_length=7, default="#FF0000", verbose_name="Color de visualización")

  triangle_type = models.CharField(max_length=20, choices=TRIANGLE_TYPES, null=True, blank=True,
                                   verbose_name="Tipo de triángulo")
  side_a = models.FloatField(null=True, blank=True, verbose_name="Lado A")
  side_b = models.FloatField(null=True, blank=True, verbose_name="Lado B")
  side_c = models.FloatField(null=True, blank=True, verbose_name="Lado C")
  inscribed_circle_diameter = models.FloatField(null=True, blank=True, verbose_name="Diámetro del círculo inscrito")
  circumscribed_circle_diameter = models.FloatField(null=True, blank=True,
                                                    verbose_name="Diámetro del círculo circunscrito")
  center_coordinates = models.JSONField(null=True, blank=True, verbose_name="Coordenadas del centro")
  is_auto_detected = models.BooleanField(default=False, db_index=True, verbose_name="Detección Automática")

  class Meta:
    ordering = ['-created_at']
    verbose_name = "Forma geométrica"
    verbose_name_plural = "Formas geométricas"

  def __str__(self):
    return f"{self.name} ({self.get_shape_type_display()})"

  def calculate_properties(self):
    from core.utils.geometry import GeometryCalculator
    calculator = GeometryCalculator(self.image)

    try:
      if self.shape_type == 'line':
        if len(self.points) != 2:
          self.properties = {'error': 'Line requires exactly 2 points'}
        else:
          self.properties = calculator.calculate_line_properties(self.points)
      elif self.shape_type == 'rectangle':
        if len(self.points) != 4:
          self.properties = {'error': 'Rectangle requires exactly 4 points'}
        else:
          self.properties = calculator.calculate_rectangle_properties(self.points)
      elif self.shape_type == 'circle':
        if len(self.points) != 2:
          self.properties = {'error': 'Circle requires exactly 2 points (center and edge)'}
        else:
          self.properties = calculator.calculate_circle_properties(self.points)
      elif self.shape_type == 'triangle':
        if len(self.points) != 3:
          self.properties = {'error': 'Triangle requires exactly 3 points'}
        else:
          props = calculator.calculate_triangle_properties(self.points)
          self.properties = props

          if 'triangle_type' in props:
            self.triangle_type = props['triangle_type']
          if 'side_lengths_px' in props and len(props['side_lengths_px']) == 3:
            self.side_a = props['side_lengths_px'][0]
            self.side_b = props['side_lengths_px'][1]
            self.side_c = props['side_lengths_px'][2]
          if 'circumscribed_circle_diameter' in props:
            self.circumscribed_circle_diameter = props['circumscribed_circle_diameter']
          if 'centroid' in props:
            self.center_coordinates = props['centroid']
      elif self.shape_type == 'polygon':
        if len(self.points) < 3:
          self.properties = {'error': 'Polygon requires at least 3 points'}
        else:
          self.properties = calculator.calculate_polygon_properties(self.points)
      else:
        self.properties = {'error': f'Unknown shape type: {self.shape_type}'}

    except Exception as e:
      self.properties = {'error': f'Calculation error: {str(e)}'}

    self.save()


class Measurement(models.Model):
  MEASUREMENT_TYPES = [
    ('distance', 'Distancia'),
    ('area', 'Área'),
    ('perimeter', 'Perímetro'),
    ('angle', 'Ángulo'),
    ('building_area', 'Área de Edificio'),
    ('land_area', 'Área de Terreno'),
    ('hectares', 'Hectáreas'),
    ('acres', 'Acres'),
    ('volume', 'Volumen'),
    ('height', 'Altura'),
    ('width', 'Ancho'),
    ('length', 'Longitud'),
  ]

  id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
  image = models.ForeignKey(ProjectImage, on_delete=models.CASCADE, related_name='measurements')
  shape = models.ForeignKey(GeometricShape, on_delete=models.CASCADE, null=True, blank=True)
  name = models.CharField(max_length=100, verbose_name="Nombre de la medición")
  measurement_type = models.CharField(max_length=20, choices=MEASUREMENT_TYPES, verbose_name="Tipo de medición")
  value_pixels = models.FloatField(verbose_name="Valor en píxeles")
  value_real = models.FloatField(null=True, blank=True, verbose_name="Valor en unidades reales")
  unit = models.CharField(max_length=20, default='px', verbose_name="Unidad de medida")
  notes = models.TextField(blank=True, verbose_name="Notas")
  created_at = models.DateTimeField(auto_now_add=True, verbose_name="Fecha de creación")

  is_manual = models.BooleanField(default=False, verbose_name="Es Manual")

  class Meta:
    ordering = ['-created_at']
    verbose_name = "Medición"
    verbose_name_plural = "Mediciones"

  def __str__(self):
    return f"{self.name}: {self.value_display}"

  @property
  def value_display(self):
    if self.value_real is not None:
      return f"{self.value_real:.2f} {self.unit}"
    return f"{self.value_pixels:.2f} px"

  def _calculate_real_value(self, scale, cal_unit):
    if self.measurement_type in ['distance', 'perimeter', 'height', 'width', 'length']:
      return self.value_pixels / scale, cal_unit
    elif self.measurement_type in ['area', 'building_area', 'land_area', 'volume']:
      return self.value_pixels / (scale ** 2), f"{cal_unit}²"
    elif self.measurement_type == 'hectares':
      area_m2 = self.value_pixels / (scale ** 2)
      if cal_unit == 'm':
        value = area_m2 / 10000
      elif cal_unit == 'cm':
        value = (area_m2 / 10000) / 10000
      elif cal_unit == 'mm':
        value = (area_m2 / 10000) / 1000000
      else:
        value = area_m2 / 10000
      return value, 'ha'
    elif self.measurement_type == 'acres':
      area_m2 = self.value_pixels / (scale ** 2)
      if cal_unit == 'm':
        value = area_m2 / 4046.86
      elif cal_unit == 'cm':
        value = (area_m2 / 4046.86) / 10000
      elif cal_unit == 'mm':
        value = (area_m2 / 4046.86) / 1000000
      else:
        value = area_m2 / 4046.86
      return value, 'ac'
    else:
      return self.value_pixels / (scale ** 2), f"{cal_unit}²"

  def save(self, *args, **kwargs):
    if (self.image.is_calibrated and
      self.image.scale_factor and
      self.value_pixels is not None):
      scale = self.image.scale_factor
      cal_unit = self.image.calibration_unit
      self.value_real, self.unit = self._calculate_real_value(scale, cal_unit)
    elif self.value_pixels is None:
      self.value_real = None

    super().save(*args, **kwargs)


class Report(models.Model):
  REPORT_FORMATS = [
    ('pdf', 'PDF'),
    ('excel', 'Excel'),
    ('json', 'JSON'),
  ]

  id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
  project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='reports')
  name = models.CharField(max_length=200, verbose_name="Nombre del reporte")
  format = models.CharField(max_length=10, choices=REPORT_FORMATS, verbose_name="Formato")
  file_path = models.CharField(max_length=500, verbose_name="Ruta del archivo")
  generated_at = models.DateTimeField(auto_now_add=True, verbose_name="Fecha de generación")
  parameters = models.JSONField(
    default=dict,
    help_text="Parámetros utilizados para generar el reporte"
  )

  class Meta:
    ordering = ['-generated_at']
    verbose_name = "Reporte"
    verbose_name_plural = "Reportes"

  def __str__(self):
    return f"{self.project.name} - {self.name}"

  @property
  def file_exists(self):
    return os.path.exists(self.file_path) if self.file_path else False


class VideoSession(models.Model):
  id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
  project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='video_sessions')
  name = models.CharField(max_length=200, verbose_name="Nombre de la sesión")
  is_active = models.BooleanField(default=False, verbose_name="Sesión activa")
  started_at = models.DateTimeField(auto_now_add=True, verbose_name="Iniciada en")
  ended_at = models.DateTimeField(null=True, blank=True, verbose_name="Terminada en")

  snapshot = models.ImageField(
    upload_to='video_snapshots/',
    null=True,
    blank=True,
    verbose_name="Captura de la sesión"
  )

  is_calibrated = models.BooleanField(default=False, verbose_name="Está calibrada")
  calibration_data = models.JSONField(
    default=dict,
    help_text="Datos de calibración para video en tiempo real"
  )

  settings = models.JSONField(
    default=dict,
    help_text="Configuraciones de la sesión de video"
  )

  class Meta:
    ordering = ['-started_at']
    verbose_name = "Sesión de Video"
    verbose_name_plural = "Sesiones de Video"

  def __str__(self):
    return f"{self.project.name} - {self.name}"

  @property
  def duration(self):
    if self.ended_at:
      return self.ended_at - self.started_at
    return None

  @property
  def is_running(self):
    return self.is_active and not self.ended_at


class VideoMeasurement(models.Model):
  MEASUREMENT_TYPES = [
    ('distance', 'Distancia'),
    ('area', 'Área'),
    ('perimeter', 'Perímetro'),
    ('angle', 'Ángulo'),
    ('height', 'Altura'),
    ('width', 'Ancho'),
    ('length', 'Longitud'),
  ]

  TRIANGLE_TYPES = [
    ('equilatero', 'Equilátero'),
    ('isosceles', 'Isósceles'),
    ('escaleno', 'Escaleno'),
  ]

  id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
  video_session = models.ForeignKey(VideoSession, on_delete=models.CASCADE, related_name='measurements')
  name = models.CharField(max_length=100, verbose_name="Nombre de la medición")
  measurement_type = models.CharField(max_length=20, choices=MEASUREMENT_TYPES, verbose_name="Tipo de medición")
  value_pixels = models.FloatField(verbose_name="Valor en píxeles")
  value_real = models.FloatField(null=True, blank=True, verbose_name="Valor en unidades reales")
  unit = models.CharField(max_length=20, default='px', verbose_name="Unidad de medida")

  shape_type = models.CharField(max_length=20, blank=True, null=True, verbose_name="Tipo de forma")
  area = models.FloatField(null=True, blank=True, verbose_name="Área (cm²)")
  perimeter = models.FloatField(null=True, blank=True, verbose_name="Perímetro (cm)")

  width = models.FloatField(null=True, blank=True, verbose_name="Ancho (cm)")
  height = models.FloatField(null=True, blank=True, verbose_name="Alto (cm)")

  radius = models.FloatField(null=True, blank=True, verbose_name="Radio (cm)")
  diameter = models.FloatField(null=True, blank=True, verbose_name="Diámetro (cm)")

  triangle_type = models.CharField(max_length=20, choices=TRIANGLE_TYPES, null=True, blank=True,
                                   verbose_name="Tipo de triángulo")
  side_a = models.FloatField(null=True, blank=True, verbose_name="Lado A (cm)")
  side_b = models.FloatField(null=True, blank=True, verbose_name="Lado B (cm)")
  side_c = models.FloatField(null=True, blank=True, verbose_name="Lado C (cm)")

  object_id = models.IntegerField(blank=True, null=True, verbose_name="ID del objeto en tiempo real")
  is_active = models.BooleanField(default=True, verbose_name="Objeto activo en la escena")

  points = models.JSONField(help_text="Puntos de medición en el video")
  frame_data = models.JSONField(
    default=dict,
    help_text="Datos del frame cuando se tomó la medición"
  )
  created_at = models.DateTimeField(auto_now_add=True, verbose_name="Fecha de creación")
  timestamp = models.FloatField(help_text="Timestamp en segundos desde el inicio de la sesión")

  class Meta:
    ordering = ['-created_at']
    verbose_name = "Medición de Video"
    verbose_name_plural = "Mediciones de Video"

  def __str__(self):
    return f"{self.name}: {self.value_display}"

  @property
  def value_display(self):
    if self.value_real is not None:
      return f"{self.value_real:.2f} {self.unit}"
    return f"{self.value_pixels:.2f} px"


class UserPreferences(models.Model):
  UNITS = [
    ('cm', 'Centímetros'),
    ('mm', 'Milímetros'),
    ('m', 'Metros'),
    ('in', 'Pulgadas'),
    ('ft', 'Pies'),
  ]

  THEMES = [
    ('light', 'Tema Claro'),
    ('dark', 'Tema Oscuro'),
  ]

  QUALITY_LEVELS = [
    ('high', 'Alta Calidad'),
    ('medium', 'Calidad Media'),
    ('low', 'Baja Calidad'),
  ]

  user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='preferences')

  default_unit = models.CharField(
    max_length=10,
    choices=UNITS,
    default='cm',
    verbose_name="Unidad de medida predeterminada"
  )

  theme = models.CharField(
    max_length=10,
    choices=THEMES,
    default='light',
    verbose_name="Tema de interfaz"
  )

  email_notifications = models.BooleanField(
    default=True,
    verbose_name="Notificaciones por email"
  )

  auto_calibration_suggestions = models.BooleanField(
    default=False,
    verbose_name="Sugerencias de calibración automática"
  )

  image_quality = models.CharField(
    max_length=10,
    choices=QUALITY_LEVELS,
    default='medium',
    verbose_name="Calidad de imagen"
  )

  auto_save = models.BooleanField(
    default=True,
    verbose_name="Auto-guardado"
  )

  auto_detection = models.BooleanField(
    default=False,
    verbose_name="Detección automática de formas"
  )

  auto_backup = models.BooleanField(
    default=False,
    verbose_name="Respaldo automático"
  )

  video_quality = models.CharField(
    max_length=10,
    choices=QUALITY_LEVELS,
    default='medium',
    verbose_name="Calidad de video"
  )

  auto_video_detection = models.BooleanField(
    default=True,
    verbose_name="Detección automática en video"
  )

  video_fps = models.IntegerField(
    default=30,
    verbose_name="FPS del video"
  )

  created_at = models.DateTimeField(auto_now_add=True, verbose_name="Fecha de creación")
  updated_at = models.DateTimeField(auto_now=True, verbose_name="Última actualización")

  class Meta:
    verbose_name = "Preferencias de Usuario"
    verbose_name_plural = "Preferencias de Usuarios"

  def __str__(self):
    return f"Preferencias de {self.user.username}"

  @classmethod
  def get_or_create_for_user(cls, user):
    preferences, created = cls.objects.get_or_create(user=user)
    return preferences
