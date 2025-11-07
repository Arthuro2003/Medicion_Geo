import math
from typing import Optional, List

from django.core.management.base import BaseCommand
from django.db import transaction

from core.models import VideoMeasurement


class Command(BaseCommand):
  help = 'Rellenar los valores de perímetro para registros de VideoMeasurement cuyo perímetro es nulo'

  def _calculate_circle_perimeter(self, radius: str) -> Optional[float]:
    try:
      return 2 * math.pi * float(radius)
    except (ValueError, TypeError):
      return None

  def _calculate_polygon_perimeter(self, sides: List[str]) -> Optional[float]:
    try:
      return sum(float(side) for side in sides if side)
    except (ValueError, TypeError):
      return None

  def _calculate_rectangle_perimeter(self, width: str, height: str) -> Optional[float]:
    try:
      return 2 * (float(width) + float(height))
    except (ValueError, TypeError):
      return None

  def _calculate_perimeter(self, measurement: VideoMeasurement) -> Optional[float]:
    if measurement.radius:
      return self._calculate_circle_perimeter(measurement.radius)

    sides = [s for s in (measurement.side_a, measurement.side_b, measurement.side_c) if s]
    if sides:
      return self._calculate_polygon_perimeter(sides)

    if measurement.width and measurement.height:
      return self._calculate_rectangle_perimeter(measurement.width, measurement.height)

    return None

  def handle(self, *args, **options):
    measurements = VideoMeasurement.objects.filter(perimeter__isnull=True)
    total = measurements.count()

    if total == 0:
      self.stdout.write(self.style.SUCCESS('No hay mediciones con perímetro nulo.'))
      return

    updated = 0
    with transaction.atomic():
      for measurement in measurements.select_for_update():
        perimeter = self._calculate_perimeter(measurement)

        if perimeter is not None:
          measurement.perimeter = perimeter
          measurement.save(update_fields=['perimeter'])
          updated += 1

    self.stdout.write(
      self.style.SUCCESS(
        f'Procesadas {total} mediciones, actualizadas {updated} con perímetro.'
      )
    )
