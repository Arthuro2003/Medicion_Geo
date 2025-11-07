from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional, Union, final

import cv2
import numpy as np
import numpy.typing as npt

try:
  from scipy.spatial.distance import pdist, squareform
  from scipy.spatial import distance

  HAS_SCIPY = True
except ImportError:
  HAS_SCIPY = False

try:
  from skimage import feature, measure, segmentation
  from skimage.feature import shape_index
  from skimage.morphology import skeletonize

  HAS_SKIMAGE = True
except ImportError:
  HAS_SKIMAGE = False

NDArray = npt.NDArray[np.float64]
Point2D = Dict[str, float]
Points = List[Point2D]


@final
@dataclass(frozen=True)
class CalibrationData:
  origin: Point2D
  scale_x: float
  scale_y: float
  rotation_rad: float
  x_real_cm: float
  y_real_cm: float
  x_point: Point2D
  y_point: Point2D


@dataclass(frozen=False)
class GeometryState:
  is_calibrated: bool = False
  transformation_matrix: Optional[NDArray] = None
  calibration_points: Points = field(default_factory=list, repr=True)
  real_distances: List[float] = field(default_factory=list, repr=True)
  calibration: Optional[CalibrationData] = None
  scale_factor: Union[Dict[str, float], float, None] = None
  calibration_unit: str = field(default='px', repr=True)
  unit: str = field(default='px', repr=True)


class GeometryCalculator:
  def __init__(self) -> None:
    self._state = GeometryState()

  def calibrate_from_points(
    self,
    pixel_points: Optional[Points] = None,
    real_distances: Optional[List[float]] = None,
    origin: Optional[Point2D] = None,
    x_point: Optional[Point2D] = None,
    y_point: Optional[Point2D] = None,
    x_real_cm: Optional[float] = None,
    y_real_cm: Optional[float] = None
  ) -> Optional[CalibrationData]:
    if pixel_points and real_distances:
      return self._legacy_calibration(pixel_points, real_distances)

    if all([origin, x_point, y_point, x_real_cm, y_real_cm]):
      return self._axis_based_calibration(origin, x_point, y_point, x_real_cm, y_real_cm)

    raise ValueError("Parámetros de calibración inválidos")

  def _legacy_calibration(self, pixel_points: Points, real_distances: List[float]) -> None:
    if len(pixel_points) != 3 or len(real_distances) != 2:
      raise ValueError("Se requieren 3 puntos de píxeles y 2 distancias reales")

    self._state.calibration_points = pixel_points
    self._state.real_distances = real_distances

    d1 = np.linalg.norm(np.array([pixel_points[0]['x'], pixel_points[0]['y']]) -
                        np.array([pixel_points[1]['x'], pixel_points[1]['y']]))
    d2 = np.linalg.norm(np.array([pixel_points[1]['x'], pixel_points[1]['y']]) -
                        np.array([pixel_points[2]['x'], pixel_points[2]['y']]))

    scale1 = real_distances[0] / d1 if d1 > 0 else 0
    scale2 = real_distances[1] / d2 if d2 > 0 else 0
    self._state.scale_factor = (scale1 + scale2) / 2.0
    self._state.is_calibrated = True
    return None

  def _axis_based_calibration(
    self,
    origin: Point2D,
    x_point: Point2D,
    y_point: Point2D,
    x_real_cm: float,
    y_real_cm: float
  ) -> CalibrationData:
    vec_x = np.array([x_point['x'] - origin['x'], x_point['y'] - origin['y']])
    vec_y = np.array([y_point['x'] - origin['x'], y_point['y'] - origin['y']])

    dist_x_px = np.linalg.norm(vec_x)
    dist_y_px = np.linalg.norm(vec_y)

    scale_x = dist_x_px / x_real_cm if x_real_cm > 0 else 0
    scale_y = dist_y_px / y_real_cm if y_real_cm > 0 else 0
    angle_rad = np.arctan2(vec_x[1], vec_x[0])

    calibration_data = CalibrationData(
      origin=origin,
      scale_x=scale_x,
      scale_y=scale_y,
      rotation_rad=angle_rad,
      x_real_cm=x_real_cm,
      y_real_cm=y_real_cm,
      x_point=x_point,
      y_point=y_point
    )

    self._state.calibration = calibration_data
    self._state.scale_factor = {'x': scale_x, 'y': scale_y}
    self._state.calibration_unit = 'cm'
    self._state.unit = 'cm'
    self._state.calibrated = True

    return calibration_data

  def pixel_to_cm(self, pt1: Union[Point2D, Tuple[float, float]],
                  pt2: Optional[Union[Point2D, Tuple[float, float]]] = None
                  ) -> Union[Dict[str, float], float]:
    if not self._state.calibrated:
      raise RuntimeError("El calculador geométrico no está calibrado")

    if pt2 is not None:
      def _to_xy(p: Union[Point2D, Tuple[float, float]]) -> Tuple[float, float]:
        if isinstance(p, dict):
          return float(p.get('x', 0)), float(p.get('y', 0))
        return float(p[0]), float(p[1])

      x1, y1 = _to_xy(pt1)
      x2, y2 = _to_xy(pt2)

      dx = x2 - x1
      dy = y2 - y1

      if isinstance(self._state.scale_factor, dict):
        angle = self._state.calibration.rotation_rad if self._state.calibration else 0.0

        rot_matrix = np.array([
          [np.cos(-angle), -np.sin(-angle)],
          [np.sin(-angle), np.cos(-angle)]
        ])

        aligned = rot_matrix @ np.array([dx, dy])

        scale_x = self._state.scale_factor.get('x', 0)
        scale_y = self._state.scale_factor.get('y', 0)

        dist_x_cm = abs(aligned[0]) / scale_x if scale_x > 0 else 0
        dist_y_cm = abs(aligned[1]) / scale_y if scale_y > 0 else 0

        return float(np.sqrt(dist_x_cm ** 2 + dist_y_cm ** 2))
      else:
        pixel_dist = np.linalg.norm(np.array([dx, dy]))
        return float(pixel_dist / (self._state.scale_factor or 1))

    elif self._state.calibration:
      px_point = pt1 if isinstance(pt1, dict) else {'x': pt1[0], 'y': pt1[1]}
      origin = self._state.calibration.origin
      scale_x = self._state.calibration.scale_x
      scale_y = self._state.calibration.scale_y
      angle = self._state.calibration.rotation_rad

      vec = np.array([px_point['x'] - origin['x'], px_point['y'] - origin['y']])

      rot_matrix = np.array([
        [np.cos(-angle), -np.sin(-angle)],
        [np.sin(-angle), np.cos(-angle)]
      ])
      aligned = rot_matrix @ vec

      cm_x = aligned[0] / scale_x if scale_x > 0 else 0
      cm_y = aligned[1] / scale_y if scale_y > 0 else 0
      return {'x_cm': cm_x, 'y_cm': cm_y}
    else:
      raise ValueError("Parámetros inválidos para la conversión de píxeles a cm")

  def is_calibrated(self) -> bool:
    return self._state.calibrated

  def cm_to_pixel(self, cm_point: Dict[str, float]) -> Point2D:
    if not self._state.calibration:
      raise ValueError('No se ha establecido la calibración')

    origin = self._state.calibration.origin
    scale_x = self._state.calibration.scale_x
    scale_y = self._state.calibration.scale_y
    angle = self._state.calibration.rotation_rad

    px_vec = np.array([cm_point['x_cm'] * scale_x, cm_point['y_cm'] * scale_y])

    rot_matrix = np.array([
      [np.cos(angle), -np.sin(angle)],
      [np.sin(angle), np.cos(angle)]
    ])
    px_aligned = rot_matrix @ px_vec

    px_x = origin['x'] + px_aligned[0]
    px_y = origin['y'] + px_aligned[1]
    return {'x': px_x, 'y': px_y}

  def area_px_to_cm2(self, area_px: float) -> float:
    if not self.is_calibrated():
      raise ValueError('No se ha establecido la calibración')

    if isinstance(self._state.scale_factor, dict):
      scale_x = self._state.scale_factor.get('x', 1)
      scale_y = self._state.scale_factor.get('y', 1)
      return area_px / (scale_x * scale_y) if (scale_x > 0 and scale_y > 0) else 0

    return area_px / (self._state.scale_factor ** 2) if self._state.scale_factor else 0

  @staticmethod
  def calculate_distance(point1: Point2D, point2: Point2D) -> float:
    dx = point2['x'] - point1['x']
    dy = point2['y'] - point1['y']
    return math.sqrt(dx * dx + dy * dy)

  def calculate_line_properties(self, points: Points) -> Dict[str, Any]:
    if len(points) != 2:
      raise ValueError("La línea requiere exactamente 2 puntos")

    distance_px = self.calculate_distance(points[0], points[1])
    dx = points[1]['x'] - points[0]['x']
    dy = points[1]['y'] - points[0]['y']
    angle = math.degrees(math.atan2(dy, dx))

    properties = {
      'length_px': distance_px,
      'angle_degrees': angle,
      'start_point': points[0],
      'end_point': points[1]
    }

    if not self._state.calibrated:
      return properties

    if isinstance(self._state.scale_factor, dict):
      scale_x = self._state.scale_factor.get('x', 1)
      scale_y = self._state.scale_factor.get('y', 1)
      if scale_x > 0 and scale_y > 0:
        dist_x_cm = abs(dx) / scale_x
        dist_y_cm = abs(dy) / scale_y
        properties['length_real'] = math.sqrt(dist_x_cm * dist_x_cm + dist_y_cm * dist_y_cm)
      else:
        properties['length_real'] = 0
    else:
      properties['length_real'] = distance_px / (self._state.scale_factor or 1)

    properties['unit'] = self._state.unit
    return properties

  def calculate_rectangle_properties(self, points: Points) -> Dict[str, Any]:
    if len(points) != 4:
      raise ValueError("El rectángulo requiere exactamente 4 puntos")

    points = self._order_rectangle_points(points)
    width_px = self.calculate_distance(points[0], points[1])
    height_px = self.calculate_distance(points[1], points[2])
    area_px = width_px * height_px
    perimeter_px = 2 * (width_px + height_px)

    centroid = {
      'x': sum(p['x'] for p in points) / 4,
      'y': sum(p['y'] for p in points) / 4
    }

    aspect_ratio = width_px / height_px if height_px > 0 else 1.0

    dx = points[1]['x'] - points[0]['x']
    dy = points[1]['y'] - points[0]['y']
    orientation = math.degrees(math.atan2(dy, dx))

    properties = {
      'width_px': width_px,
      'height_px': height_px,
      'area_px': area_px,
      'perimeter_px': perimeter_px,
      'centroid': centroid,
      'corners': points,
      'aspect_ratio': aspect_ratio,
      'orientation_degrees': orientation
    }

    if self.is_calibrated():
      if isinstance(self._state.scale_factor, dict):
        scale_x = self._state.scale_factor.get('x', 1)
        scale_y = self._state.scale_factor.get('y', 1)

        avg_scale = (scale_x + scale_y) / 2.0 if (scale_x > 0 and scale_y > 0) else 1.0

        width_real = width_px / avg_scale if avg_scale > 0 else 0
        height_real = height_px / avg_scale if avg_scale > 0 else 0

        properties.update({
          'width_real': width_real,
          'height_real': height_real,
          'area_real': self.area_px_to_cm2(area_px),
          'perimeter_real': (width_real + height_real) * 2,
          'unit': self._state.unit
        })
      else:
        scale = self._state.scale_factor or 1
        properties.update({
          'width_real': width_px / scale,
          'height_real': height_px / scale,
          'area_real': area_px / (scale ** 2),
          'perimeter_real': perimeter_px / scale,
          'unit': self._state.unit
        })

    return properties

  def calculate_circle_properties(self, points: Points) -> Dict[str, Any]:
    if len(points) != 2:
      raise ValueError("El círculo requiere 2 puntos: centro y circunferencia")

    center = points[0]
    circumference_point = points[1]

    radius_px = self.calculate_distance(center, circumference_point)
    area_px = math.pi * radius_px ** 2
    perimeter_px = 2 * math.pi * radius_px

    properties = {
      'center': center,
      'radius_px': radius_px,
      'diameter_px': 2 * radius_px,
      'area_px': area_px,
      'perimeter_px': perimeter_px
    }

    if self.is_calibrated():
      if isinstance(self._state.scale_factor, dict):
        scale_x = self._state.scale_factor.get('x', 1)
        scale_y = self._state.scale_factor.get('y', 1)
        avg_scale = (scale_x + scale_y) / 2.0 if (scale_x > 0 and scale_y > 0) else 1.0

        center_real = self.pixel_to_cm(center)

        properties.update({
          'radius_real': radius_px / avg_scale if avg_scale > 0 else 0,
          'diameter_real': 2 * radius_px / avg_scale if avg_scale > 0 else 0,
          'area_real': self.area_px_to_cm2(area_px),
          'perimeter_real': (2 * math.pi * radius_px) / avg_scale if avg_scale > 0 else 0,
          'center_real': center_real,
          'unit': self._state.unit
        })
      else:
        scale = self._state.scale_factor or 1
        properties.update({
          'radius_real': radius_px / scale,
          'diameter_real': 2 * radius_px / scale,
          'area_real': area_px / (scale ** 2),
          'perimeter_real': perimeter_px / scale,
          'unit': self._state.unit
        })

    return properties

  def calculate_triangle_properties(self, points: Points) -> Dict[str, Any]:
    if len(points) != 3:
      raise ValueError("El triángulo requiere exactamente 3 puntos")

    side_a_px = self.calculate_distance(points[0], points[1])
    side_b_px = self.calculate_distance(points[1], points[2])
    side_c_px = self.calculate_distance(points[2], points[0])

    s = (side_a_px + side_b_px + side_c_px) / 2
    area_px = math.sqrt(abs(s * (s - side_a_px) * (s - side_b_px) * (s - side_c_px)))
    perimeter_px = side_a_px + side_b_px + side_c_px

    def safe_acos(x: float) -> float:
      return math.acos(max(-1, min(1, x)))

    angle_a = math.degrees(safe_acos((side_b_px ** 2 + side_c_px ** 2 - side_a_px ** 2) / (2 * side_b_px * side_c_px)))
    angle_b = math.degrees(safe_acos((side_a_px ** 2 + side_c_px ** 2 - side_b_px ** 2) / (2 * side_a_px * side_c_px)))
    angle_c = 180 - angle_a - angle_b

    centroid = {
      'x': sum(p['x'] for p in points) / 3,
      'y': sum(p['y'] for p in points) / 3
    }

    triangle_type = self._classify_triangle(side_a_px, side_b_px, side_c_px, angle_a, angle_b, angle_c)

    circumscribed_circle_radius = (side_a_px * side_b_px * side_c_px) / (4 * area_px) if area_px > 0 else 0
    circumscribed_circle_diameter = 2 * circumscribed_circle_radius

    properties = {
      'side_lengths_px': [side_a_px, side_b_px, side_c_px],
      'area_px': area_px,
      'perimeter_px': perimeter_px,
      'angles_degrees': [angle_a, angle_b, angle_c],
      'centroid': centroid,
      'vertices': points,
      'triangle_type': triangle_type,
      'circumscribed_circle_diameter': circumscribed_circle_diameter
    }

    if self.is_calibrated():
      side_lengths_real = [
        self.pixel_to_cm(points[0], points[1]),
        self.pixel_to_cm(points[1], points[2]),
        self.pixel_to_cm(points[2], points[0])
      ]

      properties.update({
        'side_lengths_real': side_lengths_real,
        'area_real': self.area_px_to_cm2(area_px),
        'perimeter_real': sum(side_lengths_real),
        'unit': self._state.unit
      })

    return properties

  def calculate_polygon_properties(self, points: Points) -> Dict[str, Any]:
    if len(points) < 3:
      raise ValueError("El polígono requiere al menos 3 puntos")

    n = len(points)
    area_px = 0
    for i in range(n):
      j = (i + 1) % n
      area_px += points[i]['x'] * points[j]['y']
      area_px -= points[j]['x'] * points[i]['y']
    area_px = abs(area_px) / 2

    perimeter_px = 0
    perimeter_real = 0
    for i in range(n):
      j = (i + 1) % n
      perimeter_px += self.calculate_distance(points[i], points[j])
      if self.is_calibrated():
        perimeter_real += self.pixel_to_cm(points[i], points[j])

    centroid = {
      'x': sum(p['x'] for p in points) / n,
      'y': sum(p['y'] for p in points) / n
    }

    cx = cy = 0
    for i in range(n):
      j = (i + 1) % n
      factor = points[i]['x'] * points[j]['y'] - points[j]['x'] * points[i]['y']
      cx += (points[i]['x'] + points[j]['x']) * factor
      cy += (points[i]['y'] + points[j]['y']) * factor

    if area_px > 0:
      geometric_centroid = {
        'x': cx / (6 * area_px),
        'y': cy / (6 * area_px)
      }
    else:
      geometric_centroid = centroid

    is_convex = self._is_convex_polygon(points)

    x_coords = [p['x'] for p in points]
    y_coords = [p['y'] for p in points]
    bounding_box = {
      'min_x': min(x_coords),
      'max_x': max(x_coords),
      'min_y': min(y_coords),
      'max_y': max(y_coords)
    }
    bounding_box['width'] = bounding_box['max_x'] - bounding_box['min_x']
    bounding_box['height'] = bounding_box['max_y'] - bounding_box['min_y']

    properties = {
      'vertices_count': n,
      'area_px': area_px,
      'perimeter_px': perimeter_px,
      'centroid': centroid,
      'geometric_centroid': geometric_centroid,
      'vertices': points,
      'is_convex': is_convex,
      'bounding_box': bounding_box
    }

    if self.is_calibrated():
      properties.update({
        'area_real': self.area_px_to_cm2(area_px),
        'perimeter_real': perimeter_real,
        'unit': self._state.unit
      })

    return properties

  def calculate_angle_between_lines(self, line1_points: Points, line2_points: Points) -> float:
    if len(line1_points) != 2 or len(line2_points) != 2:
      raise ValueError("Cada línea debe tener exactamente 2 puntos")

    dx1 = line1_points[1]['x'] - line1_points[0]['x']
    dy1 = line1_points[1]['y'] - line1_points[0]['y']
    dx2 = line2_points[1]['x'] - line2_points[0]['x']
    dy2 = line2_points[1]['y'] - line2_points[0]['y']

    dot_product = dx1 * dx2 + dy1 * dy2
    magnitude1 = math.sqrt(dx1 ** 2 + dy1 ** 2)
    magnitude2 = math.sqrt(dx2 ** 2 + dy2 ** 2)

    if magnitude1 == 0 or magnitude2 == 0:
      return 0.0

    cos_angle = dot_product / (magnitude1 * magnitude2)
    cos_angle = max(-1, min(1, cos_angle))

    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)

    return angle_deg

  def _classify_triangle(self, a: float, b: float, c: float,
                         angle_a: float, angle_b: float, angle_c: float) -> str:
    tolerance = 1e-2

    if abs(a - b) < tolerance and abs(b - c) < tolerance:
      return "equilátero"
    elif abs(a - b) < tolerance or abs(b - c) < tolerance or abs(a - c) < tolerance:
      return "isósceles"
    else:
      return "escaleno"

  def _is_convex_polygon(self, points: Points) -> bool:
    n = len(points)
    if n < 3:
      return False

    sign = None
    for i in range(n):
      j = (i + 1) % n
      k = (i + 2) % n

      dx1 = points[j]['x'] - points[i]['x']
      dy1 = points[j]['y'] - points[i]['y']
      dx2 = points[k]['x'] - points[j]['x']
      dy2 = points[k]['y'] - points[j]['y']

      cross_product = dx1 * dy2 - dy1 * dx2

      if abs(cross_product) < 1e-10:
        continue

      current_sign = 1 if cross_product > 0 else -1

      if sign is None:
        sign = current_sign
      elif sign != current_sign:
        return False

    return True

  def _order_rectangle_points(self, points: Points) -> Points:
    pts = [(p['x'], p['y']) for p in points]

    pts.sort(key=lambda p: (p[1], p[0]))

    top_points = sorted(pts[:2], key=lambda p: p[0])
    bottom_points = sorted(pts[2:], key=lambda p: p[0])

    ordered = [
      top_points[0],
      top_points[1],
      bottom_points[1],
      bottom_points[0]
    ]

    return [{'x': p[0], 'y': p[1]} for p in ordered]


class ImageProcessor:

  def __init__(self, image: np.ndarray = None):
    self.image = image

  def set_image(self, image: np.ndarray):
    self.image = image

  def get_image(self) -> np.ndarray:
    return self.image


class AdvancedEdgeDetector:

  def __init__(self):
    self.logger = logging.getLogger('core.edge_detection')

  def enhanced_canny_detection(self, image: np.ndarray, low_threshold: int = 50,
                               high_threshold: int = 150, apertureSize: int = 3,
                               L2gradient: bool = True, blur_kernel: int = 5) -> np.ndarray:
    if len(image.shape) == 3:
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
      gray = image.copy()

    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    edges = cv2.Canny(enhanced, low_threshold, high_threshold,
                      apertureSize=apertureSize, L2gradient=L2gradient)

    return edges

  def multi_scale_edge_detection(
    self,
    image: np.ndarray,
    scales: Optional[List[float]] = None
  ) -> np.ndarray:
    if scales is None:
      scales = [1.0, 1.5, 2.0]

    if len(image.shape) == 3:
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
      gray = image.copy()

    combined_edges = np.zeros_like(gray)

    for scale in scales:
      if scale != 1.0:
        height, width = gray.shape
        new_height, new_width = int(height / scale), int(width / scale)
        scaled_img = cv2.resize(gray, (new_width, new_height))
      else:
        scaled_img = gray

      edges = self.enhanced_canny_detection(scaled_img)

      if scale != 1.0:
        edges = cv2.resize(edges, (gray.shape[1], gray.shape[0]))

      combined_edges = cv2.bitwise_or(combined_edges, edges)

    return combined_edges

  def adaptive_threshold_edges(self, image: np.ndarray, max_value: int = 255,
                               adaptive_method: int = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               threshold_type: int = cv2.THRESH_BINARY,
                               block_size: int = 11, C: int = 2) -> np.ndarray:
    if len(image.shape) == 3:
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
      gray = image.copy()

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(blurred, max_value, adaptive_method,
                                   threshold_type, block_size, C)

    edges = cv2.Canny(thresh, 50, 150)

    return edges


class ShapeContextDescriptor:

  def __init__(self, n_bins_r: int = 5, n_bins_theta: int = 12):
    self.n_bins_r = n_bins_r
    self.n_bins_theta = n_bins_theta
    self.n_bins = n_bins_r * n_bins_theta

  def compute_shape_context(self, points: np.ndarray) -> np.ndarray:
    n_points = len(points)
    shape_contexts = np.zeros((n_points, self.n_bins))

    for i in range(n_points):
      rel_coords = points - points[i]

      rel_coords = np.delete(rel_coords, i, axis=0)

      distances = np.sqrt(np.sum(rel_coords ** 2, axis=1))
      angles = np.arctan2(rel_coords[:, 1], rel_coords[:, 0])

      angles = (angles + 2 * np.pi) % (2 * np.pi)

      if len(distances) > 0:
        r_bins = np.logspace(np.log10(np.min(distances) + 1e-6),
                             np.log10(np.max(distances) + 1e-6),
                             self.n_bins_r + 1)
        theta_bins = np.linspace(0, 2 * np.pi, self.n_bins_theta + 1)

        r_indices = np.digitize(distances, r_bins) - 1
        theta_indices = np.digitize(angles, theta_bins) - 1

        r_indices = np.clip(r_indices, 0, self.n_bins_r - 1)
        theta_indices = np.clip(theta_indices, 0, self.n_bins_theta - 1)

        for r_idx, theta_idx in zip(r_indices, theta_indices):
          bin_idx = r_idx * self.n_bins_theta + theta_idx
          shape_contexts[i, bin_idx] += 1

    for i in range(n_points):
      total = np.sum(shape_contexts[i])
      if total > 0:
        shape_contexts[i] /= total

    return shape_contexts

  def match_shapes(self, shape1_points: np.ndarray, shape2_points: np.ndarray,
                   threshold: float = 0.3) -> Dict[str, Any]:
    sc1 = self.compute_shape_context(shape1_points)
    sc2 = self.compute_shape_context(shape2_points)

    n1, n2 = len(sc1), len(sc2)
    cost_matrix = np.zeros((n1, n2))

    for i in range(n1):
      for j in range(n2):
        cost_matrix[i, j] = self._chi_square_distance(sc1[i], sc2[j])

    matches = []
    matched_indices = set()

    for i in range(n1):
      best_j = np.argmin(cost_matrix[i])
      if best_j not in matched_indices and cost_matrix[i, best_j] < threshold:
        matches.append((i, best_j, cost_matrix[i, best_j]))
        matched_indices.add(best_j)

    if matches:
      total_cost = sum(match[2] for match in matches)
      avg_cost = total_cost / len(matches)
      similarity = max(0, 1 - avg_cost)
    else:
      similarity = 0

    return {
      'matches': matches,
      'similarity': similarity,
      'cost_matrix': cost_matrix,
      'n_matches': len(matches)
    }

  def _chi_square_distance(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
    epsilon = 1e-10
    return 0.5 * np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + epsilon))
