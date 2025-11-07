import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np


@dataclass
class EdgeDetectionConfig:
  LOW_THRESHOLD: int = 50
  HIGH_THRESHOLD: int = 150
  APERTURE_SIZE: int = 3
  BLUR_KERNEL: int = 5
  USE_L2_GRADIENT: bool = True
  MIN_CONTOUR_AREA_PERCENT: float = 0.01
  CLAHE_CLIP_LIMIT: float = 2.0
  CLAHE_GRID_SIZE: Tuple[int, int] = (8, 8)
  BILATERAL_FILTER_D: int = 9
  BILATERAL_SIGMA: float = 75.0
  POLYGON_EPSILON_FACTOR: float = 0.02
  REGULAR_POLYGON_TOLERANCE: float = 0.15
  DEFAULT_SCALES: Tuple[float, ...] = (1.0, 1.5, 2.0)
  ENHANCEMENT_TYPES: Tuple[str, ...] = ('adaptive', 'clahe', 'histogram', 'none')


class AdvancedEdgeDetector:

  def __init__(self):
    self.logger = logging.getLogger('core.edge_detection')
    self.config = EdgeDetectionConfig()

  def enhanced_canny_detection(
    self,
    image: np.ndarray,
    low_threshold: int = EdgeDetectionConfig.LOW_THRESHOLD,
    high_threshold: int = EdgeDetectionConfig.HIGH_THRESHOLD,
    aperture_size: int = EdgeDetectionConfig.APERTURE_SIZE,
    l2_gradient: bool = EdgeDetectionConfig.USE_L2_GRADIENT,
    blur_kernel: int = EdgeDetectionConfig.BLUR_KERNEL
  ) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    clahe = cv2.createCLAHE(clipLimit=self.config.CLAHE_CLIP_LIMIT, tileGridSize=self.config.CLAHE_GRID_SIZE)
    enhanced = clahe.apply(blurred)
    return cv2.Canny(enhanced, low_threshold, high_threshold, apertureSize=aperture_size, L2gradient=l2_gradient)

  def multi_scale_edge_detection(
    self,
    image: np.ndarray,
    scales: Tuple[float, ...] | None = None
  ) -> np.ndarray:
    if scales is None:
      scales = self.config.DEFAULT_SCALES
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    combined_edges = np.zeros_like(gray)

    for scale in scales:
      if scale == 1.0:
        scaled_img = gray
      else:
        height, width = gray.shape
        new_dims = (int(width / scale), int(height / scale))
        scaled_img = cv2.resize(gray, new_dims)

      edges = self.enhanced_canny_detection(scaled_img)

      if scale != 1.0:
        edges = cv2.resize(edges, (width, height))

      combined_edges = cv2.bitwise_or(combined_edges, edges)

    return combined_edges


@dataclass
class ShapeDetectionConfig:
  MIN_AREA_PERCENT: float = 0.01
  EPSILON_FACTOR: float = 0.02
  SQUARE_ASPECT_RATIO_MIN: float = 0.7
  SQUARE_ASPECT_RATIO_MAX: float = 1.3
  MIN_SOLIDITY: float = 0.9
  CIRCLE_CIRCULARITY: float = 0.7
  MAX_SHAPE_COUNT: int = 3
  SIZE_DIFFERENCE_THRESHOLD: float = 0.3
  KERNEL_SIZE: int = 3
  CANNY_LOW: int = 50
  CANNY_HIGH: int = 150
  BLUR_SIZE: int = 5


class ImageProcessor:
  def __init__(self):
    self.edge_detector = AdvancedEdgeDetector()
    self.logger = logging.getLogger('core.image_processing')
    self.config = ShapeDetectionConfig()

  def detect_shapes(self, image_path: str, use_advanced_edges: bool = True) -> List[Dict[str, Any]]:
    image = cv2.imread(image_path)
    if image is None:
      raise ValueError(f"No se pudo cargar la imagen desde {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    image_area = height * width

    edges = (self.edge_detector.multi_scale_edge_detection(image) if use_advanced_edges
             else cv2.Canny(
      cv2.GaussianBlur(gray, (self.config.BLUR_SIZE, self.config.BLUR_SIZE), 0),
      self.config.CANNY_LOW,
      self.config.CANNY_HIGH))

    kernel = np.ones((self.config.KERNEL_SIZE, self.config.KERNEL_SIZE), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shape_candidates = []

    for i, contour in enumerate(contours):
      area = cv2.contourArea(contour)
      if area < image_area * self.config.MIN_AREA_PERCENT:
        continue

      epsilon = self.config.EPSILON_FACTOR * cv2.arcLength(contour, True)
      approx = cv2.approxPolyDP(contour, epsilon, True)
      shape_info = self._classify_enhanced_shape(approx, contour, area)

      if shape_info:
        shape_info.update({
          'contour_id': i,
          'original_contour_area': area
        })
        shape_candidates.append(shape_info)

    shape_candidates.sort(key=lambda x: (x['original_contour_area'], x.get('circularity', 0)), reverse=True)

    circles = [s for s in shape_candidates if s['shape_type'] == 'circle']
    other_shapes = [s for s in shape_candidates if s['shape_type'] != 'circle']
    detected_shapes = []

    if circles:
      detected_shapes.append(circles[0])
      size_threshold = circles[0]['original_contour_area'] * self.config.SIZE_DIFFERENCE_THRESHOLD
      detected_shapes.extend(
        circle for circle in circles[1:]
        if abs(circle['original_contour_area'] - circles[0]['original_contour_area']) > size_threshold
      )
    elif other_shapes:
      detected_shapes.append(other_shapes[0])

    return detected_shapes[:self.config.MAX_SHAPE_COUNT]

  def _classify_enhanced_shape(self, approx: np.ndarray, contour: np.ndarray, area: float) -> Dict[str, Any]:
    vertices = len(approx)
    points = [{'x': float(pt[0][0]), 'y': float(pt[0][1])} for pt in approx]
    perimeter = cv2.arcLength(contour, True)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h if h > 0 else 1
    rect_area = w * h
    extent = area / rect_area if rect_area > 0 else 0

    shape_info = {
      'points': points,
      'contour_area': area,
      'contour_perimeter': perimeter,
      'solidity': solidity,
      'aspect_ratio': aspect_ratio,
      'extent': extent,
      'bounding_rect': {'x': x, 'y': y, 'width': w, 'height': h}
    }

    if vertices == 3:
      shape_info['shape_type'] = 'triangle'
      return shape_info

    if vertices == 4:
      shape_info['shape_type'] = ('square' if (self.config.SQUARE_ASPECT_RATIO_MIN <= aspect_ratio <=
                                               self.config.SQUARE_ASPECT_RATIO_MAX and
                                               solidity > self.config.MIN_SOLIDITY)
                                  else 'rectangle')
      return shape_info

    if vertices > 4:
      circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

      if circularity > self.config.CIRCLE_CIRCULARITY and solidity > self.config.MIN_SOLIDITY:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        shape_info.update({
          'shape_type': 'circle',
          'points': [
            {'x': float(x), 'y': float(y)},
            {'x': float(x + radius), 'y': float(y)}
          ],
          'circularity': circularity
        })
        return shape_info

      if vertices <= 8 and solidity > self.config.MIN_SOLIDITY:
        shape_info['shape_type'] = (f'regular_{vertices}_gon'
                                    if self._is_regular_polygon(approx)
                                    else 'polygon')
        return shape_info

      shape_info['shape_type'] = 'irregular_shape'
      return shape_info

    return None

  def _is_regular_polygon(self, approx: np.ndarray) -> bool:
    if len(approx) < 3:
      return False

    side_lengths = []
    for i in range(len(approx)):
      j = (i + 1) % len(approx)
      p1 = approx[i][0]
      p2 = approx[j][0]
      length = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
      side_lengths.append(length)

    mean_length = np.mean(side_lengths)
    return ((np.std(side_lengths) / mean_length) < self.config.REGULAR_POLYGON_TOLERANCE
            if mean_length > 0 else False)

  def enhance_image_for_detection(
    self,
    image_path: str,
    enhancement_type: str = 'adaptive'
  ) -> str:
    if enhancement_type not in EdgeDetectionConfig.ENHANCEMENT_TYPES:
      raise ValueError(f"Tipo de mejora no válido {enhancement_type}")

    image = cv2.imread(image_path)
    if image is None:
      raise ValueError(f"No se pudo cargar la imagen desde {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    match enhancement_type:
      case 'adaptive':
        clahe = cv2.createCLAHE(clipLimit=self.config.CLAHE_CLIP_LIMIT * 1.5,
                                tileGridSize=self.config.CLAHE_GRID_SIZE)
        enhanced = cv2.bilateralFilter(
          clahe.apply(gray),
          self.config.BILATERAL_FILTER_D,
          self.config.BILATERAL_SIGMA,
          self.config.BILATERAL_SIGMA
        )
      case 'clahe':
        clahe = cv2.createCLAHE(clipLimit=self.config.CLAHE_CLIP_LIMIT,
                                tileGridSize=self.config.CLAHE_GRID_SIZE)
        enhanced = clahe.apply(gray)
      case 'histogram':
        enhanced = cv2.equalizeHist(gray)
      case _:
        enhanced = gray

    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]], dtype=np.float32)
    enhanced = cv2.filter2D(enhanced, -1, kernel)

    enhanced_path = str(Path(image_path).with_stem(Path(image_path).stem + '_enhanced'))
    cv2.imwrite(enhanced_path, enhanced)
    return enhanced_path
