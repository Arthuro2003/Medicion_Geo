from django.contrib import admin
from django.urls import reverse
from django.utils.html import format_html

from .models import Project, ProjectImage, GeometricShape, Measurement


@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
  list_display = ['name', 'created_by', 'created_at', 'total_images', 'total_measurements', 'is_active']
  list_filter = ['is_active', 'created_at', 'updated_at']
  search_fields = ['name', 'description', 'created_by__username']
  readonly_fields = ['id', 'created_at', 'updated_at']
  date_hierarchy = 'created_at'

  fieldsets = (
    (None, {
      'fields': ('name', 'description', 'created_by', 'is_active')
    }),
    ('Información del Sistema', {
      'fields': ('id', 'created_at', 'updated_at'),
      'classes': ('collapse',)
    })
  )

  def total_images(self, obj):
    return obj.total_images

  total_images.short_description = 'Imágenes'

  def total_measurements(self, obj):
    return obj.total_measurements

  total_measurements.short_description = 'Mediciones'


class GeometricShapeInline(admin.TabularInline):
  model = GeometricShape
  extra = 0
  readonly_fields = ['id', 'properties_display', 'created_at']
  fields = ['name', 'shape_type', 'color', 'properties_display', 'created_at']

  def properties_display(self, obj):
    if not obj.properties:
      return "Sin propiedades"

    props = []
    for key, value in obj.properties.items():
      if isinstance(value, (int, float)):
        props.append(f"{key}: {value:.2f}")
      else:
        props.append(f"{key}: {value}")

    return " | ".join(props[:3]) + ("..." if len(props) > 3 else "")

  properties_display.short_description = 'Propiedades'


class MeasurementInline(admin.TabularInline):
  model = Measurement
  extra = 0
  readonly_fields = ['id', 'value_display', 'created_at']
  fields = ['name', 'measurement_type', 'value_display', 'unit', 'created_at']


@admin.register(ProjectImage)
class ProjectImageAdmin(admin.ModelAdmin):
  list_display = ['name', 'project', 'dimensions', 'is_calibrated', 'measurements_count', 'shapes_count', 'uploaded_at']
  list_filter = ['is_calibrated', 'uploaded_at', 'project']
  search_fields = ['name', 'project__name']
  readonly_fields = ['id', 'width', 'height', 'uploaded_at', 'image_preview']
  date_hierarchy = 'uploaded_at'

  fieldsets = (
    (None, {
      'fields': ('project', 'name', 'image', 'image_preview')
    }),
    ('Información de la Imagen', {
      'fields': ('width', 'height', 'uploaded_at')
    }),
    ('Calibración', {
      'fields': ('is_calibrated',),
      'classes': ('collapse',)
    }),
    ('Información del Sistema', {
      'fields': ('id',),
      'classes': ('collapse',)
    })
  )

  inlines = [GeometricShapeInline, MeasurementInline]

  def dimensions(self, obj):
    return f"{obj.width} × {obj.height} px"

  dimensions.short_description = 'Dimensiones'

  def measurements_count(self, obj):
    count = obj.measurements.count()
    if count > 0:
      url = reverse('admin:core_measurement_changelist') + f'?image__id__exact={obj.id}'
      return format_html('<a href="{}">{} mediciones</a>', url, count)
    return '0 mediciones'

  measurements_count.short_description = 'Mediciones'

  def shapes_count(self, obj):
    count = obj.shapes.count()
    if count > 0:
      url = reverse('admin:core_geometricshape_changelist') + f'?image__id__exact={obj.id}'
      return format_html('<a href="{}">{} formas</a>', url, count)
    return '0 formas'

  shapes_count.short_description = 'Formas'

  def image_preview(self, obj):
    if obj.image:
      return format_html(
        '<img src="{}" style="max-width: 300px; max-height: 200px; object-fit: contain;" />',
        obj.image.url
      )
    return "Sin propiedades de imagen"
