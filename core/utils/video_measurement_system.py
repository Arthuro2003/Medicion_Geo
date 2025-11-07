import json
import math
import os
from collections import deque

import cv2
import numpy as np
from django.conf import settings
from scipy.spatial import distance as dist


class RealTimeMeasurementSystem:

  def __init__(self, aruco_size_cm=5.0):
    self.aruco_size_cm = aruco_size_cm

    self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    self.aruco_params = cv2.aruco.DetectorParameters()
    self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

    self.ratio_history = deque(maxlen=10)

    self.tracked_objects = {}
    self.next_object_id = 0
    self.MAX_DISAPPEARED_FRAMES = 15
    self.MAX_TRACKING_DISTANCE = 75

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

    self.config_file = os.path.join(settings.MEDIA_ROOT, 'measurement_config.json')
    self.load_config()

  def load_config(self):
    if os.path.exists(self.config_file):
      try:
        with open(self.config_file, 'r') as f:
          saved_config = json.load(f)
          self.config.update(saved_config.get('config', {}))
          self.aruco_size_cm = saved_config.get('aruco_size_cm', self.aruco_size_cm)
      except Exception as e:
        print(f"Error al cargar configuración: {e}")

  def save_config(self):
    try:
      data = {
        'config': self.config,
        'aruco_size_cm': self.aruco_size_cm
      }
      os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
      with open(self.config_file, 'w') as f:
        json.dump(data, f)
    except Exception as e:
      print(f"Error al guardar configuración: {e}")

  def detectar_aruco_con_perspectiva(self, img):
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

  def detectar_forma_robusta(self, contour):
    peri = cv2.arcLength(contour, True)
    if peri == 0: return "Objeto", None

    area = cv2.contourArea(contour)
    circularity = 4 * math.pi * area / (peri * peri)

    if circularity >= self.config['CIRCULARITY_THRESHOLD']: return "Circulo", None

    approx = cv2.approxPolyDP(contour, self.config['APPROX_EPS_FACTOR'] * peri, True)
    vertices = len(approx)

    if 3 <= vertices <= 5:
      if vertices == 3: return "Triangulo", approx
      if vertices == 4:
        _, (w, h), _ = cv2.minAreaRect(contour)
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        return "Cuadrado" if aspect_ratio <= 1.2 else "Rectangulo", approx

    return "Objeto", None

  def detectar_objetos_robustos(self, img, pixels_per_cm, aruco_corners=None):
    h_img, w_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 80, 80)

    binary_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)

    if aruco_corners is not None:
      exclude_mask = np.zeros_like(gray)
      expanded_corners = aruco_corners + (aruco_corners - np.mean(aruco_corners, axis=0)) * 0.15
      cv2.fillPoly(exclude_mask, [expanded_corners.astype(int)], 255)
      binary_img = cv2.bitwise_and(binary_img, cv2.bitwise_not(exclude_mask))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.config['MORPH_KERNEL'], self.config['MORPH_KERNEL']))
    cleaned = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    for c in contours:
      area_px = cv2.contourArea(c)
      if not (self.config['ABS_MIN_AREA_PX'] < area_px < w_img * h_img * 0.5): continue

      hull = cv2.convexHull(c)
      hull_area = cv2.contourArea(hull)
      if hull_area > 0:
        solidity = area_px / hull_area
        if solidity < self.config['MIN_SOLIDITY']: continue

      _, (w, h), _ = cv2.minAreaRect(c)
      if w == 0 or h == 0: continue
      aspect_ratio = max(w, h) / min(w, h)
      if aspect_ratio > self.config['MAX_ASPECT_RATIO']: continue

      valid_contours.append(c)
    return valid_contours

  def register_object(self, centroid, contour):
    self.tracked_objects[self.next_object_id] = {
      'centroid': centroid,
      'contour': contour,
      'disappeared': 0,
      'latest_approx': None,
      'shape_history': deque(maxlen=self.config['MEASUREMENT_HISTORY_SIZE']),
      'measurements': {
        key: deque(maxlen=self.config['MEASUREMENT_HISTORY_SIZE'])
        for key in ['area_cm2', 'perim_cm', 'radius_cm', 'diameter_cm', 'width_cm', 'height_cm', 'sides_cm']
      }
    }
    self.next_object_id += 1

  def deregister_object(self, object_id):
    if object_id in self.tracked_objects:
      del self.tracked_objects[object_id]

  def update_tracked_objects(self, contours):
    input_centroids = np.zeros((len(contours), 2), dtype="int")
    for i, c in enumerate(contours):
      M = cv2.moments(c)
      if M["m00"] == 0: continue
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
    cols = D.argmin(axis=1)[rows]

    used_rows = set()
    used_cols = set()

    for (row, col) in zip(rows, cols):
      if row in used_rows or col in used_cols:
        continue

      if D[row, col] > self.MAX_TRACKING_DISTANCE:
        continue

      object_id = object_ids[row]
      self.tracked_objects[object_id].update({
        'centroid': input_centroids[col],
        'contour': contours[col],
        'disappeared': 0
      })

      used_rows.add(row)
      used_cols.add(col)

    unused_rows = set(range(object_centroids.shape[0])) - used_rows
    for row in unused_rows:
      object_id = object_ids[row]
      self.tracked_objects[object_id]['disappeared'] += 1
      if self.tracked_objects[object_id]['disappeared'] > self.MAX_DISAPPEARED_FRAMES:
        self.deregister_object(object_id)

    unused_cols = set(range(input_centroids.shape[0])) - used_cols
    for col in unused_cols:
      self.register_object(input_centroids[col], contours[col])

  def _calculate_perimeter_for_non_circle(self, w_px, h_px, pixels_per_cm):
    try:
      w_cm = (w_px / pixels_per_cm)
      h_cm = (h_px / pixels_per_cm)
      return 2 * (w_cm + h_cm)
    except Exception:
      return None

  def calcular_y_almacenar_mediciones(self, object_id, pixels_per_cm):
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
      try:
        radius_cm = (radius_px / pixels_per_cm)
        obj['measurements']['perim_cm'].append(2 * math.pi * radius_cm)
      except Exception:
        pass
    elif shape_name == "Triangulo" and approx is not None and len(approx) == 3:
      vertices = approx.reshape(-1, 2)
      sides_cm = [np.linalg.norm(vertices[i] - vertices[(i + 1) % 3]) / pixels_per_cm for i in range(3)]
      obj['measurements']['sides_cm'].append(sides_cm)
      obj['measurements']['perim_cm'].append(sum(sides_cm))
    else:
      _, (w_px, h_px), _ = cv2.minAreaRect(contour)
      obj['measurements']['width_cm'].append(w_px / pixels_per_cm)
      obj['measurements']['height_cm'].append(h_px / pixels_per_cm)
      perimeter = self._calculate_perimeter_for_non_circle(w_px, h_px, pixels_per_cm)
      if perimeter is not None:
        obj['measurements']['perim_cm'].append(perimeter)

  def _get_stable_average(self, measurements_deque):
    return np.mean(measurements_deque) if measurements_deque else 0

  def _get_stable_sides(self, sides_deque):
    return np.mean(sides_deque, axis=0) if sides_deque else [0, 0, 0]

  def _draw_label_background(self, display_frame, cx, cy, text_size, color):
    text_w, text_h = text_size
    bg_coords = (
      (int(cx - text_w / 2 - 5), int(cy - text_h - 5)),
      (int(cx + text_w / 2 + 5), int(cy + 5))
    )
    overlay = display_frame.copy()
    cv2.rectangle(overlay, bg_coords[0], bg_coords[1], (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, display_frame, 0.5, 0, display_frame)
    return display_frame

  def _draw_text(self, display_frame, text, cx, cy, text_size, color):
    text_w, text_h = text_size
    cv2.putText(display_frame, text,
                (int(cx - text_w / 2), int(cy + text_h / 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return display_frame

  def _draw_shape_outline(self, display_frame, stable_shape, latest_approx, contour, color):
    if stable_shape in ["Triangulo", "Rectangulo", "Cuadrado"]:
      if latest_approx is not None:
        cv2.polylines(display_frame, [latest_approx], True, color, 2)
      else:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        cv2.polylines(display_frame, [np.int32(box)], True, color, 2)
    return display_frame

  def dibujar_mediciones_estabilizadas(self, img, pixels_per_cm, aruco_corners=None):
    if pixels_per_cm is None or pixels_per_cm <= 0:
      return img.copy(), []

    display_frame = img.copy()
    measurements = []

    if aruco_corners is not None:
      cv2.polylines(display_frame, [aruco_corners.astype(int)], True, (0, 255, 0), 2)

    colors = [(100, 255, 100), (255, 150, 100), (150, 150, 255),
              (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    for i, object_id in enumerate(list(self.tracked_objects.keys())):
      if object_id not in self.tracked_objects:
        continue

      obj = self.tracked_objects[object_id]
      self.calcular_y_almacenar_mediciones(object_id, pixels_per_cm)

      stable_shape = "..."
      if obj['shape_history']:
        stable_shape = max(set(obj['shape_history']), key=list(obj['shape_history']).count)

      def get_stable_avg(key):
        return self._get_stable_average(obj['measurements'][key])

      def get_stable_sides():
        return self._get_stable_sides(obj['measurements']['sides_cm'])

      color = colors[i % len(colors)]
      cv2.drawContours(display_frame, [obj['contour']], -1, color, 2)

      measurement = {
        'object_id': i + 1,
        'shape_type': stable_shape,
        'area_cm2': get_stable_avg('area_cm2'),
        'perimeter_cm': get_stable_avg('perim_cm'),
        'width_cm': get_stable_avg('width_cm'),
        'height_cm': get_stable_avg('height_cm'),
        'radius_cm': get_stable_avg('radius_cm'),
        'diameter_cm': get_stable_avg('diameter_cm'),
        'is_active': True
      }

      if stable_shape == "Triangulo":
        stable_sides = get_stable_sides()
        if len(stable_sides) == 3:
          measurement.update({
            'sides_cm': stable_sides,
            'side_a': stable_sides[0],
            'side_b': stable_sides[1],
            'side_c': stable_sides[2]
          })
      elif stable_shape == "Circulo":
        measurement.update({
          'radius': get_stable_avg('radius_cm'),
          'diameter': get_stable_avg('diameter_cm')
        })
      measurements.append(measurement)

      obj_label = f"Obj: {i + 1}"
      cx, cy = obj['centroid']

      text_size, _ = cv2.getTextSize(obj_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
      display_frame = self._draw_label_background(display_frame, cx, cy, text_size, color)
      display_frame = self._draw_text(display_frame, obj_label, cx, cy, text_size, color)

      display_frame = self._draw_shape_outline(display_frame, stable_shape, obj['latest_approx'], obj['contour'], color)

    return display_frame, measurements

  def process_frame(self, frame):
    _, pixel_cm_ratio, aruco_corners = self.detectar_aruco_con_perspectiva(frame)
    display_frame = frame.copy()
    measurements = []

    if pixel_cm_ratio is not None:
      contours = self.detectar_objetos_robustos(display_frame, pixel_cm_ratio, aruco_corners)
      self.update_tracked_objects(contours)
      display_frame, measurements = self.dibujar_mediciones_estabilizadas(
        display_frame, pixel_cm_ratio, aruco_corners)
    else:
      self.tracked_objects.clear()

    return display_frame, measurements
