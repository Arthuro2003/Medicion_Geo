import math
import queue
import threading
import time
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np

from .geometry import GeometryCalculator
from .video_measurement_system import RealTimeMeasurementSystem


class VideoProcessor:

  def __init__(self):
    self.cap = None
    self.measurement_system = RealTimeMeasurementSystem(aruco_size_cm=5.0)
    self.current_frame = None
    self.measurement_points = []
    self.geometry_calculator = GeometryCalculator()
    self.reference_length = None
    self.detected_shapes = []
    self.gesture_mode = False
    self.calibration_mode = False
    self.measurement_mode = False
    self.gesture_controller = None
    self.auto_calibrator = None
    self.gesture_tracker = None

  def initialize_camera(self, camera_index: int = 0) -> bool:
    try:
      self.cap = cv2.VideoCapture(camera_index)
      if not self.cap.isOpened():
        return False

      self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
      self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
      self.cap.set(cv2.CAP_PROP_FPS, 30)

      return True
    except Exception as e:
      print(f"Error al inicializar cámara: {e}")
      return False

  def release_camera(self):
    try:
      if self.cap and self.cap.isOpened():
        self.cap.release()
    except Exception as e:
      print(f"Advertencia: Error al liberar cámara: {e}")
    finally:
      self.cap = None

  def capture_frame(self) -> Optional[np.ndarray]:
    if not self.cap or not self.cap.isOpened():
      return None

    ret, frame = self.cap.read()
    if ret:
      self.current_frame = frame.copy()
      return frame
    return None

  def _draw_measurement_points(self, frame: np.ndarray) -> np.ndarray:
    if self.measurement_points:
      for i, point in enumerate(self.measurement_points):
        cv2.circle(frame, point, 5, (0, 255, 0), -1)
        cv2.putText(frame, f"P{i + 1}",
                    (point[0] + 10, point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

      if len(self.measurement_points) == 2:
        cv2.line(frame,
                 self.measurement_points[0],
                 self.measurement_points[1],
                 (0, 255, 0), 2)

        p1 = np.array(self.measurement_points[0])
        p2 = np.array(self.measurement_points[1])
        pixel_distance = np.linalg.norm(p2 - p1)

        mid_point = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        distance_text = f"Distancia: {pixel_distance:.1f} px"
        cv2.putText(frame, distance_text,
                    (mid_point[0] - 80, mid_point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame

  def process_frame(self, frame: np.ndarray) -> Dict:
    if frame is None:
      return {}

    processed_frame, measurements = self.measurement_system.process_frame(frame)
    processed_frame = self._draw_measurement_points(processed_frame)

    results = {
      'frame': processed_frame,
      'measurements': measurements,
    }

    return results

  def add_measurement_point(self, x: int, y: int):
    if len(self.measurement_points) < 2:
      self.measurement_points.append((x, y))
    else:
      self.measurement_points[0] = (x, y)

  def clear_measurement_points(self):
    self.measurement_points = []

  def get_frame_with_measurements(self) -> Optional[np.ndarray]:
    if self.current_frame is not None:
      return self.process_frame(self.current_frame)['frame']
    return None

  def save_measurement_data(self) -> Dict:
    data = {
      'measurement_points': self.measurement_points,
      'reference_length': self.reference_length,
      'detected_shapes': self.detected_shapes,
      'timestamp': cv2.getTickCount() / cv2.getTickFrequency()
    }
    return data

  def detect_shapes_in_frame(self, frame: np.ndarray) -> List[Dict]:
    try:
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      blur = cv2.bilateralFilter(gray, 9, 75, 75)

      thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)

      kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
      opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
      closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

      edges = cv2.Canny(closed, 50, 150)

      contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

      shapes: List[Dict] = []
      for contour in contours:
        if cv2.contourArea(contour) > 400:
          shape_info = self._detect_shape_from_contour(contour)
          if shape_info:
            try:
              shape_info['contour'] = contour.tolist()
            except Exception:
              shape_info['contour'] = []
            shapes.append(shape_info)

      circles = self._detect_circles_in_frame(gray, edges)
      for c in circles:
        shapes.append(c)

      return shapes
    except Exception as e:
      print(f"Error al detectar formas: {e}")
      return []

  def _detect_shape_from_contour(self, contour):
    try:
      epsilon = 0.02 * cv2.arcLength(contour, True)
      approx = cv2.approxPolyDP(contour, epsilon, True)

      area = float(cv2.contourArea(contour))
      perimeter = float(cv2.arcLength(contour, True))
      vertices = int(len(approx))

      center = None
      radius = None
      shape_type = 'unknown'

      if vertices == 3:
        shape_type = 'triangle'
      elif vertices == 4:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / float(h) if h != 0 else 0
        shape_type = 'square' if 0.95 <= aspect_ratio <= 1.05 else 'rectangle'
      elif vertices > 4:
        if perimeter > 0:
          circularity = 4 * math.pi * area / (perimeter * perimeter)
        else:
          circularity = 0

        if circularity > 0.70:
          (x, y), r = cv2.minEnclosingCircle(contour)
          center = (float(x), float(y))
          radius = float(r)
          shape_type = 'circle'
        else:
          shape_type = 'polygon'
      else:
        shape_type = 'unknown'

      return {
        'type': shape_type,
        'area': area,
        'perimeter': perimeter,
        'vertices': vertices,
        'center': center,
        'radius': radius
      }
    except Exception as e:
      print(f"Error al detectar forma del contorno: {e}")
      return None

  def _detect_circles_in_frame(self, gray_frame: np.ndarray, edges: Optional[np.ndarray] = None) -> List[Dict]:
    shapes: List[Dict] = []
    try:
      blurred = cv2.medianBlur(gray_frame, 5)
      if edges is None:
        edges = cv2.Canny(blurred, 50, 150)
      circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                                 param1=60, param2=45, minRadius=12, maxRadius=0)
      if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
          y0, x0, rr = int(y), int(x), int(r)
          h, w = blurred.shape[:2]
          if x0 - rr < 0 or x0 + rr >= w or y0 - rr < 0 or y0 + rr >= h:
            continue
          angles = np.linspace(0, 2 * math.pi, num=24)
          support = 0
          for a in angles:
            sx = int(x0 + rr * math.cos(a))
            sy = int(y0 + rr * math.sin(a))
            if 0 <= sx < w and 0 <= sy < h:
              if edges[sy, sx] > 0:
                support += 1
          if support < 6:
            continue
          shapes.append({
            'type': 'circle',
            'area': math.pi * (float(r) ** 2),
            'perimeter': 2 * math.pi * float(r),
            'vertices': 0,
            'center': (float(x), float(y)),
            'radius': float(r),
            'contour': []
          })
    except Exception:
      pass
    return shapes

  def _get_shape_display_name(self, shape_type: str) -> str:
    mapping = {
      'triangle': 'Triángulo',
      'square': 'Cuadrado',
      'rectangle': 'Rectángulo',
      'circle': 'Círculo',
      'polygon': 'Polígono',
      'unknown': 'Desconocido'
    }
    return mapping.get(shape_type, shape_type)

  def _calculate_real_dimensions(self, shape: Dict, width_px: float, height_px: float) -> Tuple[
    float, float, float, float, str, str]:
    if self.geometry_calculator.is_calibrated():
      unit = 'cm'
      area_unit = 'cm2'
      try:
        b0 = np.array([shape.get('contour', [[]])[0][0], shape.get('contour', [[]])[0][1]]) if shape.get(
          'contour') else np.array([0, 0])
        b1 = np.array([shape.get('contour', [[]])[1][0], shape.get('contour', [[]])[1][1]]) if len(
          shape.get('contour', [[]])) > 1 else np.array([width_px, 0])
        b2 = np.array([shape.get('contour', [[]])[2][0], shape.get('contour', [[]])[2][1]]) if len(
          shape.get('contour', [[]])) > 2 else np.array([width_px, height_px])
        corner_p1 = {'x': float(b0[0]), 'y': float(b0[1])}
        corner_p2 = {'x': float(b1[0]), 'y': float(b1[1])}
        corner_p3 = {'x': float(b2[0]), 'y': float(b2[1])}

        width_real = self.geometry_calculator.pixel_to_cm(corner_p1, corner_p2)
        height_real = self.geometry_calculator.pixel_to_cm(corner_p2, corner_p3)
        area_real = self.area_px_to_cm2(float(shape.get('area', 0)))
        perimeter_real = width_real * 2 + height_real * 2
      except Exception:
        width_real = self.pixel_to_cm((0, 0), (width_px, 0))
        height_real = self.pixel_to_cm((0, 0), (0, height_px))
        try:
          area_real = self.area_px_to_cm2(float(shape.get('area', 0)))
        except Exception:
          area_real = 0.0
        perimeter_real = (width_real + height_real) * 2
    else:
      unit = 'px'
      area_unit = 'px2'
      width_real = float(width_px)
      height_real = float(height_px)
      area_real = shape.get('area', 0)
      perimeter_real = shape.get('perimeter', 0)
    return width_real, height_real, area_real, perimeter_real, unit, area_unit

  def _draw_shape_text(self, frame: np.ndarray, text: str, position: Tuple[int, int], font_scale: float = 0.5,
                       color: Tuple[int, int, int] = (255, 100, 0), thickness: int = 1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, position, font, font_scale, color, thickness)

  def draw_shape_annotations(self, frame: np.ndarray, shape: Dict):
    try:
      raw_contour = shape.get('contour', [])
      if raw_contour is None:
        raw_contour = []
      contour = np.array(raw_contour, dtype=np.int32)
      if contour.size == 0:
        return
      if contour.ndim == 3 and contour.shape[1] == 1:
        contour = contour.reshape((-1, 2))
      if contour.ndim != 2 or contour.shape[1] != 2 or len(contour) < 3:
        return

      try:
        rect = cv2.minAreaRect(contour.astype(np.float32))
        box = cv2.boxPoints(rect)
        box = np.int0(box)
      except Exception:
        x, y, w, h = cv2.boundingRect(contour)
        rect = ((x + w / 2.0, y + h / 2.0), (w, h), 0.0)
        box = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)

      cv2.drawContours(frame, [box], 0, (255, 100, 0), 2)

      width_px = rect[1][0]
      height_px = rect[1][1]

      try:
        b0 = box[0].astype(float)
        b1 = box[1].astype(float)
        b2 = box[2].astype(float)
        width_px_exact = float(np.linalg.norm(b1 - b0))
        height_px_exact = float(np.linalg.norm(b2 - b1))
        if width_px_exact >= height_px_exact:
          width_px = width_px_exact
          height_px = height_px_exact
        else:
          width_px = height_px_exact
          height_px = width_px_exact
      except Exception:
        try:
          width_px = float(rect[1][0])
          height_px = float(rect[1][1])
        except Exception:
          width_px = 0.0
          height_px = 0.0

      width_real, height_real, area_real, perimeter_real, unit, area_unit = self._calculate_real_dimensions(shape,
                                                                                                            width_px,
                                                                                                            height_px)

      font = cv2.FONT_HERSHEY_SIMPLEX
      font_scale = 0.5
      font_color = (255, 100, 0)
      thickness = 1

      center = rect[0]
      width_text = f"{width_real:.2f} {unit}"
      height_text = f"{height_real:.2f} {unit}"

      try:
        sorted_by_y = sorted(box.tolist(), key=lambda p: p[1])
        v1, v2, v3, v4 = map(np.array, sorted_by_y)
        if v1[0] > v2[0]:
          v1, v2 = v2, v1
        if v4[0] > v3[0]:
          v3, v4 = v4, v3
      except Exception:
        x, y, w, h = cv2.boundingRect(contour)
        v1 = np.array([x, y])
        v2 = np.array([x + w, y])
        v3 = np.array([x + w, y + h])
        v4 = np.array([x, y + h])

      mid_top = ((v1[0] + v2[0]) // 2, (v1[1] + v2[1]) // 2)
      mid_left = ((v1[0] + v4[0]) // 2, (v1[1] + v4[1]) // 2)

      try:
        self._draw_shape_text(frame, self._get_shape_display_name(shape['type']), (int(center[0]), int(center[1])), 0.7,
                              font_color, 2)
      except Exception:
        pass
      self._draw_shape_text(frame, width_text, (mid_top[0] - 20, mid_top[1] - 10), font_scale, font_color, thickness)
      self._draw_shape_text(frame, height_text, (mid_left[0] - 80, mid_left[1]), font_scale, font_color, thickness)

      info_text = f"Área: {area_real:.2f} {area_unit} | Perímetro: {perimeter_real:.2f} {unit}"
      self._draw_shape_text(frame, info_text, (box[0][0], box[0][1] - 40), font_scale, font_color, thickness)

    except Exception as e:
      print(f"Error al dibujar anotaciones de forma: {e}")

    try:
      if shape.get('type') == 'circle' and shape.get('center') and shape.get('radius'):
        c = shape['center']
        r_px = float(shape['radius'])
        try:
          cv2.circle(frame, (int(c[0]), int(c[1])), int(r_px), (0, 200, 200), 2)
        except Exception:
          pass

        if self.geometry_calculator.is_calibrated():
          try:
            center_pt = {'x': float(c[0]), 'y': float(c[1])}
            circ_pt = {'x': float(c[0]) + r_px, 'y': float(c[1])}
            r_real = self.geometry_calculator.pixel_to_cm(center_pt, circ_pt)
            area_real = math.pi * (r_real ** 2)
            perimeter_real = 2 * math.pi * r_real
            unit = getattr(self.geometry_calculator, 'unit', 'cm')
            area_unit = unit + '2'
            self._draw_shape_text(frame, f"Radio: {r_real:.2f} {unit}", (int(c[0]) + 10, int(c[1]) + 10), 0.5,
                                  (0, 200, 200), 2)
            self._draw_shape_text(frame, f"Área: {area_real:.2f} {area_unit}", (int(c[0]) + 10, int(c[1]) + 30), 0.5,
                                  (0, 200, 200), 2)
            self._draw_shape_text(frame, f"Perímetro: {perimeter_real:.2f} {unit}", (int(c[0]) + 10, int(c[1]) + 50),
                                  0.5, (0, 200, 200), 2)
          except Exception:
            pass
    except Exception:
      pass

  def set_reference_length(self, length: float):
    self.reference_length = length

  def set_gesture_mode(self, enabled: bool):
    self.gesture_mode = enabled

  def set_calibration_mode(self, enabled: bool):
    self.calibration_mode = enabled
    self.gesture_controller.set_calibration_mode(enabled)

  def set_measurement_mode(self, enabled: bool):
    self.measurement_mode = enabled
    self.gesture_controller.set_measurement_mode(enabled)

  def auto_calibrate_with_fingers(self, frame: np.ndarray, known_distance_cm: float) -> Dict:
    return self.auto_calibrator.calibrate_with_fingers(frame, known_distance_cm)

  def get_gesture_measurement_points(self) -> List[Tuple[int, int]]:
    return self.gesture_controller.get_measurement_points()

  def get_gesture_calibration_distance(self) -> Optional[float]:
    return self.gesture_controller.get_calibration_distance()

  def clear_gesture_data(self):
    self.gesture_controller.clear_calibration()
    self.gesture_controller.clear_measurements()

  def is_auto_calibrated(self) -> bool:
    return self.auto_calibrator.is_calibrated()

  def get_auto_calibration_info(self) -> Dict:
    return self.auto_calibrator.get_calibration_info()

  def _draw_real_time_measurements(self, frame: np.ndarray):
    if not self.gesture_tracker.is_ready_for_measurement():
      return

    measurements = self.gesture_tracker.get_current_measurements()
    if not measurements:
      return

    points = self.gesture_tracker.current_points
    shape_type = measurements.get('type', 'unknown')
    is_calibrated = 'width_cm' in measurements or 'distance_cm' in measurements
    unit = 'cm' if is_calibrated else 'px'
    area_unit = 'cm2' if is_calibrated else 'px2'

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (0, 255, 255)
    thickness = 2

    if shape_type == 'line' and len(points) == 2:
      p1, p2 = points
      cv2.line(frame, p1, p2, font_color, thickness)

      val = measurements.get('distance_cm', measurements.get('distance', 0))
      text = f"{val:.2f} {unit}"
      midpoint = measurements.get('midpoint', (0, 0))
      cv2.putText(frame, text, (int(midpoint[0]) - 40, int(midpoint[1]) - 20), font, font_scale, font_color, thickness)

    elif shape_type == 'rectangle' and len(points) >= 2:
      x_coords = [p[0] for p in points]
      y_coords = [p[1] for p in points]
      min_x, max_x = min(x_coords), max(x_coords)
      min_y, max_y = min(y_coords), max(y_coords)

      cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), font_color, thickness)

      width = measurements.get('width_cm', measurements.get('width', 0))
      height = measurements.get('height_cm', measurements.get('height', 0))
      width_text = f"{width:.2f} {unit}"
      height_text = f"{height:.2f} {unit}"
      cv2.putText(frame, width_text, (min_x, min_y - 10), font, font_scale, font_color, thickness)
      cv2.putText(frame, height_text, (max_x + 10, min_y + (max_y - min_y) // 2), font, font_scale, font_color,
                  thickness)

      area = measurements.get('area_cm2', measurements.get('area', 0))
      perimeter = measurements.get('perimeter_cm', measurements.get('perimeter', 0))
      info_text = f"Área: {area:.2f} {area_unit} | Perímetro: {perimeter:.2f} {unit}"
      cv2.putText(frame, info_text, (min_x, max_y + 25), font, font_scale, font_color, thickness)

    elif shape_type in ('triangle', 'polygon') and len(points) >= 3:
      pts = np.array(points, np.int32)
      cv2.polylines(frame, [pts], True, font_color, thickness)

      area = measurements.get('area_cm2', measurements.get('area', 0))
      perimeter = measurements.get('perimeter_cm', measurements.get('perimeter', 0))
      info_text = f"Área: {area:.2f} {area_unit} | Perímetro: {perimeter:.2f} {unit}"
      center = measurements.get('center', (0, 0))
      cv2.putText(frame, info_text, (int(center[0]) - 80, int(center[1])), font, font_scale, font_color, thickness)

  def get_real_time_measurements(self) -> Dict:
    return self.gesture_tracker.get_current_measurements()

  def get_measurement_summary(self) -> str:
    return self.gesture_tracker.get_measurement_summary()

  def clear_gesture_measurements(self):
    self.gesture_tracker.clear_points()


class VideoStreamHandler:

  def __init__(self):
    self.cap = None
    self.is_streaming = False
    self.frame_queue = queue.Queue(maxsize=2)
    self.processed_queue = queue.Queue(maxsize=2)
    self.measurement_system = RealTimeMeasurementSystem(aruco_size_cm=5.0)

    self.frame_width = 1280
    self.frame_height = 720
    self.fps = 30

    self.current_measurements = []
    self._measurement_lock = threading.Lock()
    self.measurement_points = []

  def _draw_measurement_points(self, frame: np.ndarray) -> np.ndarray:
    if self.measurement_points:
      for i, point in enumerate(self.measurement_points):
        cv2.circle(frame, point, 5, (0, 255, 0), -1)
        cv2.putText(frame, f"P{i + 1}", (point[0] + 10, point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

      if len(self.measurement_points) == 2:
        cv2.line(frame, self.measurement_points[0],
                 self.measurement_points[1], (0, 255, 0), 2)
    return frame

  def start_stream(self, camera_index: int = 0) -> bool:
    try:
      self.cap = cv2.VideoCapture(camera_index)
      if not self.cap.isOpened():
        raise ValueError(f"No se pudo abrir la cámara {camera_index}")

      self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
      self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
      self.cap.set(cv2.CAP_PROP_FPS, self.fps)

      self.is_streaming = True

      capture_thread = threading.Thread(target=self._capture_loop)
      process_thread = threading.Thread(target=self._process_loop)

      capture_thread.daemon = True
      process_thread.daemon = True

      capture_thread.start()
      process_thread.start()

      return True

    except Exception as e:
      print(f"Error al iniciar stream: {e}")
      self.stop_stream()
      return False

  def stop_stream(self):
    self.is_streaming = False
    if self.cap:
      self.cap.release()
      self.cap = None

    for q in [self.frame_queue, self.processed_queue]:
      while not q.empty():
        try:
          q.get_nowait()
        except queue.Empty:
          break

  def _capture_loop(self):
    while self.is_streaming and self.cap and self.cap.isOpened():
      if not self.frame_queue.full():
        ret, frame = self.cap.read()
        if not ret:
          self.stop_stream()
          break

        try:
          self.frame_queue.put(frame, timeout=0.1)
        except queue.Full:
          continue
      else:
        time.sleep(0.01)

  def _process_loop(self):
    while self.is_streaming:
      try:
        frame = self.frame_queue.get(timeout=0.1)
      except queue.Empty:
        continue

      processed_frame, measurements = self.measurement_system.process_frame(frame)
      processed_frame = self._draw_measurement_points(processed_frame)

      with self._measurement_lock:
        self.current_measurements = measurements

      if not self.processed_queue.full():
        try:
          self.processed_queue.put(processed_frame, timeout=0.1)
        except queue.Full:
          continue

      self.frame_queue.task_done()

  def get_frame_bytes(self) -> Optional[bytes]:
    try:
      frame = self.processed_queue.get_nowait()
      self.processed_queue.task_done()
      if frame is not None:
        _, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()
    except queue.Empty:
      pass
    return None

  def process_click(self, x: int, y: int):
    if len(self.measurement_points) < 2:
      self.measurement_points.append((x, y))
    else:
      self.measurement_points = [(x, y)]

  def clear_measurements(self):
    self.measurement_points = []
    with self._measurement_lock:
      self.current_measurements = []

  def get_measurements(self):
    with self._measurement_lock:
      return self.current_measurements.copy()

  def set_calibration_data(self, cal_data: Dict):
    if cal_data and isinstance(cal_data.get('aruco_size_cm'), (int, float)):
      self.measurement_system.aruco_size_cm = float(cal_data['aruco_size_cm'])
      self.measurement_system.save_config()

  def get_calibration_info(self):
    return {
      'aruco_size_cm': self.measurement_system.aruco_size_cm,
      'config': self.measurement_system.config
    }

  def update_measurement_config(self, config: Dict):
    if isinstance(config, dict):
      self.measurement_system.config.update(config)
      self.measurement_system.save_config()
