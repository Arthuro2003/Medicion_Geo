import json
from pathlib import Path
from typing import Dict, Any

import matplotlib
from django.conf import settings

matplotlib.use('Agg')
import logging


class ReportTemplate:
  def __init__(self, template_name: str, template_config: Dict[str, Any]) -> None:
    self.name = template_name
    self.config = template_config
    self.logger = logging.getLogger('core.report_templates')

    self.default_config: Dict[str, Any] = {
      'page_size': 'A4',
      'orientation': 'portrait',
      'margins': {
        'top': 2.5,
        'bottom': 2.5,
        'left': 2.5,
        'right': 2.5
      },
      'font_family': 'Helvetica',
      'title_style': {
        'fontSize': 18,
        'textColor': '#2c3e50',
        'alignment': 'center',
        'spaceAfter': 30
      },
      'header_style': {
        'fontSize': 14,
        'textColor': '#34495e',
        'spaceAfter': 12
      },
      'body_style': {
        'fontSize': 10,
        'textColor': '#2c3e50',
        'spaceAfter': 6
      },
      'table_style': {
        'fontSize': 9,
        'headerBackground': '#3498db',
        'headerTextColor': '#ffffff',
        'alternateRowColor': '#f8f9fa',
        'borderColor': '#dee2e6'
      },
      'chart_style': {
        'width': 400,
        'height': 300,
        'colors': [
          '#3498db',
          '#e74c3c',
          '#2ecc71',
          '#f39c12',
          '#9b59b6'
        ]
      },
      'logo': {
        'enabled': True,
        'path': None,
        'width': 100,
        'height': 50,
        'alignment': 'left'
      },
      'footer': {
        'enabled': True,
        'text': 'Generado por MediciónGeo - {datetime}',
        'fontSize': 8,
        'textColor': '#7f8c8d'
      }
    }

    self.effective_config = self._merge_config(self.default_config, self.config)

  def _merge_config(self, default: Dict[str, Any], custom: Dict[str, Any]) -> Dict[str, Any]:
    result = default.copy()
    for key, value in custom.items():
      if key in result and isinstance(result[key], dict) and isinstance(value, dict):
        result[key] = self._merge_config(result[key], value)
      else:
        result[key] = value
    return result


class CustomReportBuilder:
  def __init__(self, project: Any, template_name: str = 'default') -> None:
    self.project = project
    self.template_name = template_name
    self.template = self._load_template(template_name)
    self.charts: list = []
    self.custom_sections: list = []

    self.output_dir = Path(settings.MEDIA_ROOT) / 'reports' / str(project.id)
    self.output_dir.mkdir(parents=True, exist_ok=True)

  def _load_template(self, template_name: str) -> ReportTemplate:
    template_path = Path(settings.BASE_DIR) / 'core' / 'templates' / 'reports' / f'{template_name}.json'

    if template_path.exists():
      with open(template_path, 'r') as f:
        config = json.load(f)
    else:
      config = {}

    return ReportTemplate(template_name, config)

  def add_chart(
    self, chart_type: str, data: Dict[str, Any],
    title: str, position: str = 'body'
  ) -> 'CustomReportBuilder':
    chart_config = {
      'type': chart_type,
      'data': data,
      'title': title,
      'position': position,
      'style': self.template.effective_config['chart_style'].copy()
    }

    self.charts.append(chart_config)
    return self

  def add_measurement_summary_chart(self) -> 'CustomReportBuilder':
    measurement_types: Dict[str, int] = {}
    for image in self.project.images.all():
      for measurement in image.measurements.all():
        mtype = measurement.get_measurement_type_display()
        measurement_types[mtype] = measurement_types.get(mtype, 0) + 1

    if measurement_types:
      self.add_chart(
        chart_type='pie',
        data=measurement_types,
        title='Distribución de Tipos de Medición'
      )

    return self

  def add_calibration_status_chart(self) -> 'CustomReportBuilder':
    calibrated = 0
    not_calibrated = 0

    for image in self.project.images.all():
      if image.is_calibrated:
        calibrated += 1
      else:
        not_calibrated += 1

    if calibrated + not_calibrated > 0:
      self.add_chart(
        chart_type='pie',
        data={
          'Calibradas': calibrated,
          'Sin Calibrar': not_calibrated
        },
        title='Estado de Calibración de Imágenes'
      )
