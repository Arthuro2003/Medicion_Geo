import json
import math
import os
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from django.conf import settings
from scipy.spatial import distance as dist


class RealTimeMeasurement:
  def __init__(self, aruco_size_cm: float = 5.0) -> None:
    self.aruco_size_cm = aruco_size_cm

    self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    self.aruco_params = cv2.aruco.DetectorParameters()
    self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

    self.ratio_history = deque(maxlen=10)

    self.tracked_objects: Dict[int, Dict] = {}
    self.next_object_id: int = 0
    self.MAX_DISAPPEARED_FRAMES: int = 15
    self.MAX_TRACKING_DISTANCE: int = 75

    self.config = {
      'MIN_CONTOUR_AREA_PX': 1000,
      'ABS_MIN_AREA_PX': 1200,
      'BLUR_KERNEL': 5,
      'MORPH_KERNEL': 5,
      'APPROX_EPS_FACTOR': 0.02,
      'CIRCULARITY_THRESHOLD': 0.8,
      'MIN_SOLIDITY': 0.4,
      'MAX_ASPECT_RATIO': 15.0,
      'MIN_AREA_CM2': 1.0,
      'MAX_AREA_CM2': 250.0,
      'MEASUREMENT_HISTORY_SIZE': 8
    }

    self.calibration_file = os.path.join(settings.MEDIA_ROOT, 'calibration.json')
    self.load_calibration()

  def load_calibration(self) -> None:
    if os.path.exists(self.calibration_file):
      try:
        with open(self.calibration_file, 'r') as f:
          data = json.load(f)
          self.config.update(data.get('config', {}))
          self.aruco_size_cm = data.get('aruco_size_cm', self.aruco_size_cm)
      except Exception as e:
        print(f"Error al cargar la calibración: {e}")

  def save_calibration(self) -> None:
    try:
      data = {
        'config': self.config,
        'aruco_size_cm': self.aruco_size_cm
      }
      os.makedirs(os.path.dirname(self.calibration_file), exist_ok=True)
      with open(self.calibration_file, 'w') as f:
        json.dump(data, f)
    except Exception as e:
      print(f"Error al guardar la calibración: {e}")

  def detectar_aruco_con_perspectiva(self, img: np.ndarray) -> Tuple[
    Optional[np.ndarray], Optional[float], Optional[np.ndarray]]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners_list, ids, _ = self.detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
      return None, None, None

    corners = corners_list[0].reshape((4, 2)).astype(np.float32)
    lado_promedio = np.mean([np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)])

    pixels_per_cm = lado_promedio / self.aruco_size_cm
    self.ratio_history.append(pixels_per_cm)
    stable_ratio = np.median(list(self.ratio_history))

    return img, stable_ratio, corners

  def detectar_forma_robusta(self, contour: np.ndarray) -> Tuple[str, Optional[np.ndarray]]:
    peri = cv2.arcLength(contour, True)
    if peri == 0:
      return "Objeto", None

    area = cv2.contourArea(contour)
    circularity = 4 * math.pi * area / (peri * peri)

    if circularity >= self.config['CIRCULARITY_THRESHOLD']:
      return "Circulo", None

    approx = cv2.approxPolyDP(contour, self.config['APPROX_EPS_FACTOR'] * peri, True)
    vertices = len(approx)

    if 3 <= vertices <= 5:
      if vertices == 3:
        return "Triangulo", approx
      if vertices == 4:
        _, (w, h), _ = cv2.minAreaRect(contour)
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        return "Cuadrado" if aspect_ratio <= 1.2 else "Rectangulo", approx

    return "Objeto", None

  def detectar_objetos_robustos(self, img: np.ndarray, pixels_per_cm: float,
                                aruco_corners: Optional[np.ndarray] = None) -> List[np.ndarray]:
    h_img, w_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 80, 80)

    binary_img = cv2.adaptiveThreshold(
      gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
      cv2.THRESH_BINARY_INV, 15, 4
    )

    if aruco_corners is not None:
      exclude_mask = np.zeros_like(gray)
      expanded_corners = aruco_corners + (aruco_corners - np.mean(aruco_corners, axis=0)) * 0.15
      cv2.fillPoly(exclude_mask, [expanded_corners.astype(int)], 255)
      binary_img = cv2.bitwise_and(binary_img, cv2.bitwise_not(exclude_mask))

    kernel = cv2.getStructuringElement(
      cv2.MORPH_RECT,
      (self.config['MORPH_KERNEL'], self.config['MORPH_KERNEL'])
    )
    cleaned = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    for c in contours:
      area_px = cv2.contourArea(c)
      if not (self.config['ABS_MIN_AREA_PX'] < area_px < w_img * h_img * 0.5):
        continue

      hull = cv2.convexHull(c)
      hull_area = cv2.contourArea(hull)
      if hull_area > 0:
        solidity = area_px / hull_area
        if solidity < self.config['MIN_SOLIDITY']:
          continue

      _, (w, h), _ = cv2.minAreaRect(c)
      if w == 0 or h == 0:
        continue
      aspect_ratio = max(w, h) / min(w, h)
      if aspect_ratio > self.config['MAX_ASPECT_RATIO']:
        continue

      valid_contours.append(c)
    return valid_contours

  def update_tracked_objects(self, contours: List[np.ndarray]) -> None:
    input_centroids = np.zeros((len(contours), 2), dtype="int")
    for i, c in enumerate(contours):
      M = cv2.moments(c)
      if M["m00"] == 0:
        continue
      input_centroids[i] = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    if len(contours) == 0:
      for object_id in list(self.tracked_objects.keys()):
        self.tracked_objects[object_id]['disappeared'] += 1
        if self.tracked_objects[object_id]['disappeared'] > self.MAX_DISAPPEARED_FRAMES:
          self.deregister_object(object_id)
      return

    if len(self.tracked_objects) == 0:
      for i in range(len(input_centroids)):
        self.register_object(input_centroids[i], contours[i])
      return

    object_ids = list(self.tracked_objects.keys())
    object_centroids = np.array([o['centroid'] for o in self.tracked_objects.values()])
    D = dist.cdist(object_centroids, input_centroids)
    rows = D.min(axis=1).argsort()
    cols = D.argmin(axis=1)[D.min(axis=1).argsort()]

    used_rows, used_cols = set(), set()
    for row, col in zip(rows, cols):
      if (row in used_rows or col in used_cols or
        D[row, col] > self.MAX_TRACKING_DISTANCE):
        continue

      object_id = object_ids[row]
      self.tracked_objects[object_id].update({
        'centroid': input_centroids[col],
        'contour': contours[col],
        'disappeared': 0
      })
      used_rows.add(row)
      used_cols.add(col)

    unused_rows = set(range(object_centroids.shape[0])).difference(used_rows)
    for row in unused_rows:
      object_id = object_ids[row]
      self.tracked_objects[object_id]['disappeared'] += 1
      if self.tracked_objects[object_id]['disappeared'] > self.MAX_DISAPPEARED_FRAMES:
        self.deregister_object(object_id)

    new_cols = set(range(input_centroids.shape[0])).difference(used_cols)
    for col in new_cols:
      self.register_object(input_centroids[col], contours[col])

  def register_object(self, centroid: Tuple[int, int], contour: np.ndarray) -> None:
    self.tracked_objects[self.next_object_id] = {
      'centroid': centroid,
      'contour': contour,
      'disappeared': 0,
      'latest_approx': None,
      'shape_history': deque(maxlen=self.config['MEASUREMENT_HISTORY_SIZE']),
      'measurements': {
        key: deque(maxlen=self.config['MEASUREMENT_HISTORY_SIZE'])
        for key in ['area_cm2', 'perim_cm', 'radius_cm', 'diameter_cm',
                    'width_cm', 'height_cm', 'sides_cm']
      }
    }
    self.next_object_id += 1

  def deregister_object(self, object_id: int) -> None:
    if object_id in self.tracked_objects:
      del self.tracked_objects[object_id]

  def calcular_y_almacenar_mediciones(self, object_id: int, pixels_per_cm: float) -> None:
    obj = self.tracked_objects[object_id]
    contour = obj['contour']

    shape_name, approx = self.detectar_forma_robusta(contour)
    obj['shape_history'].append(shape_name)
    obj['latest_approx'] = approx

    area_px = cv2.contourArea(contour)
    obj['measurements']['area_cm2'].append(area_px / (pixels_per_cm ** 2))

    if shape_name == "Circulo":
      _, radius_px = cv2.minEnclosingCircle(contour)
      obj['measurements']['radius_cm'].append(radius_px / pixels_per_cm)
      obj['measurements']['diameter_cm'].append((2 * radius_px) / pixels_per_cm)
    elif shape_name == "Triangulo" and approx is not None and len(approx) == 3:
      vertices = approx.reshape(-1, 2)
      sides_cm = [np.linalg.norm(vertices[i] - vertices[(i + 1) % 3]) / pixels_per_cm
                  for i in range(3)]
      obj['measurements']['sides_cm'].append(sides_cm)
      obj['measurements']['perim_cm'].append(sum(sides_cm))
    else:
      _, (w_px, h_px), _ = cv2.minAreaRect(contour)
      obj['measurements']['width_cm'].append(w_px / pixels_per_cm)
      obj['measurements']['height_cm'].append(h_px / pixels_per_cm)

  def dibujar_mediciones_estabilizadas(self, img: np.ndarray, pixels_per_cm: float,
                                       aruco_corners: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[Dict]]:
    if pixels_per_cm is None or pixels_per_cm <= 0:
      return img.copy(), []

    display_frame = img.copy()
    if aruco_corners is not None:
      cv2.polylines(display_frame, [aruco_corners.astype(int)], True, (0, 255, 0), 2)

    colors = [(100, 255, 100), (255, 150, 100), (150, 150, 255),
              (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    measurements: List[Dict] = []

    def get_stable_avg(measurements_dict: Dict, key: str) -> float:
      return np.mean(measurements_dict[key]) if measurements_dict[key] else 0

    def get_stable_sides(measurements_dict: Dict) -> np.ndarray:
      return (np.mean(measurements_dict['sides_cm'], axis=0)
              if measurements_dict['sides_cm'] else np.zeros(3))

    for i, object_id in enumerate(list(self.tracked_objects.keys())):
      if object_id not in self.tracked_objects:
        continue

      obj = self.tracked_objects[object_id]
      self.calcular_y_almacenar_mediciones(object_id, pixels_per_cm)

      stable_shape = ("..." if not obj['shape_history'] else
                      max(set(obj['shape_history']), key=list(obj['shape_history']).count))

      color = colors[i % len(colors)]
      cv2.drawContours(display_frame, [obj['contour']], -1, color, 2)

      measurement = {
        'shape_type': stable_shape,
        'area_cm2': get_stable_avg(obj['measurements'], 'area_cm2'),
        'width_cm': get_stable_avg(obj['measurements'], 'width_cm'),
        'height_cm': get_stable_avg(obj['measurements'], 'height_cm'),
        'radius_cm': get_stable_avg(obj['measurements'], 'radius_cm'),
        'diameter_cm': get_stable_avg(obj['measurements'], 'diameter_cm')
      }

      perimeter = 0.0
      if stable_shape == "Circulo":
        radius = get_stable_avg(obj['measurements'], 'radius_cm')
        perimeter = 2 * math.pi * radius
      elif stable_shape == "Triangulo":
        sides = get_stable_sides(obj['measurements'])
        perimeter = sum(sides) if len(sides) == 3 else 0.0
      else:
        width = get_stable_avg(obj['measurements'], 'width_cm')
        height = get_stable_avg(obj['measurements'], 'height_cm')
        perimeter = 2 * (width + height)

      measurement['perimeter_cm'] = perimeter
      measurements.append(measurement)

      lines = []
      if stable_shape == "Circulo":
        lines = [
          f"Forma: {stable_shape}",
          f"Radio: {get_stable_avg(obj['measurements'], 'radius_cm'):.1f} cm",
          f"Diámetro: {get_stable_avg(obj['measurements'], 'diameter_cm'):.1f} cm",
          f"Área: {get_stable_avg(obj['measurements'], 'area_cm2'):.1f} cm²"
        ]
      elif stable_shape == "Triangulo":
        stable_sides = get_stable_sides(obj['measurements'])
        if len(stable_sides) == 3:
          lines = [
            f"Forma: {stable_shape}",
            f"Lado A: {stable_sides[0]:.1f} cm",
            f"Lado B: {stable_sides[1]:.1f} cm",
            f"Lado C: {stable_sides[2]:.1f} cm",
            f"Área: {get_stable_avg(obj['measurements'], 'area_cm2'):.1f} cm²"
          ]
      else:
        formato = stable_shape if stable_shape in ["Cuadrado", "Rectangulo"] else "Objeto"
        lines = [
          f"Forma: {formato}",
          f"Ancho: {get_stable_avg(obj['measurements'], 'width_cm'):.1f} cm",
          f"Alto: {get_stable_avg(obj['measurements'], 'height_cm'):.1f} cm",
          f"Área: {get_stable_avg(obj['measurements'], 'area_cm2'):.1f} cm²"
        ]

      if lines:
        _, top_y, _, _ = cv2.boundingRect(obj['contour'])
        cx, _ = obj['centroid']
        y_pos = top_y - 15
        if y_pos - (len(lines) * 20) < 0:
          y_pos = top_y + 30 + (len(lines) * 20)
        for line in reversed(lines):
          text_size, _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
          box_coords = (
            (cx - 10, y_pos - text_size[1] - 5),
            (cx + text_size[0] + 10, y_pos + 5)
          )
          cv2.rectangle(display_frame, box_coords[0], box_coords[1],
                        (0, 0, 0), cv2.FILLED)
          cv2.putText(display_frame, line, (cx, y_pos),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
          y_pos -= 22

    return display_frame, measurements

  def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
    _, pixel_cm_ratio, aruco_corners = self.detectar_aruco_con_perspectiva(frame)
    display_frame = frame.copy()
    measurements: List[Dict] = []

    if pixel_cm_ratio is not None:
      contours = self.detectar_objetos_robustos(display_frame, pixel_cm_ratio, aruco_corners)
      self.update_tracked_objects(contours)
      display_frame, measurements = self.dibujar_mediciones_estabilizadas(
        display_frame, pixel_cm_ratio, aruco_corners
      )

      cv2.putText(
        display_frame,
        f"ArUco detectado - Ratio: {pixel_cm_ratio:.2f} px/cm",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
      )
      cv2.putText(
        display_frame,
        f"Objetos detectados: {len(self.tracked_objects)}",
        (10, display_frame.shape[0] - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
      )
      cv2.putText(
        display_frame,
        "CALIBRADO",
        (10, display_frame.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
      )
    else:
      self.tracked_objects.clear()
      cv2.putText(
        display_frame,
        "Coloque marcador ArUco (5x5 cm)",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
      )
      cv2.putText(
        display_frame,
        "ESPERANDO CALIBRACIÓN",
        (10, display_frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
      )

    return display_frame, measurements


def main():
  measurer = RealTimeMeasurement()

  cap = cv2.VideoCapture(0)

  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

  while True:
    ret, frame = cap.read()
    if not ret:
      break

    processed, measurements = measurer.process_frame(frame)

    cv2.imshow('Real-time Measurement', processed)

    if cv2.waitKey(1) & 0xFF == 27:
      break

  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()
