import math
import os
from datetime import datetime
from typing import Dict, Optional

import cv2
import numpy as np
from django.conf import settings

DEFAULT_CONFIG = {
  'MIN_CONTOUR_AREA_PX': 2000,
  'MORPH_KERNEL': 7,
  'APPROX_EPS_FACTOR': 0.02,
  'CIRCULARITY_THRESHOLD': 0.85,
  'MIN_SOLIDITY': 0.35,
  'MAX_ASPECT_RATIO': 20.0,
}

SHAPE_TYPES = {
  'TRIANGLE': 'Triángulo',
  'SQUARE': 'Cuadrado',
  'RECTANGLE': 'Rectángulo',
  'CIRCLE': 'Círculo',
  'OBJECT': 'Objeto',
}

TRIANGLE_TYPES = {
  'EQUILATERAL': 'Equilátero',
  'ISOSCELES': 'Isósceles',
  'SCALENE': 'Escaleno',
}


def ensure_dir(path: str) -> None:
  d = os.path.dirname(path)
  if d and not os.path.exists(d):
    os.makedirs(d, exist_ok=True)


def detect_and_annotate(image_path: str, aruco_real_size_cm: float = 5.0,
                        config: Optional[Dict] = None) -> Dict:
  if config is None:
    config = {
      'MIN_CONTOUR_AREA_PX': 2000,
      'MORPH_KERNEL': 7,
      'APPROX_EPS_FACTOR': 0.02,
      'CIRCULARITY_THRESHOLD': 0.85,
      'MIN_SOLIDITY': 0.35,
      'MAX_ASPECT_RATIO': 20.0,
    }

  img = cv2.imread(image_path)
  if img is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

  h_img, w_img = img.shape[:2]

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  try:
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners_list, ids, _ = detector.detectMarkers(gray)
  except Exception as e:
    corners_list, ids = None, None

  if ids is None or len(ids) == 0:
    raise RuntimeError('No ArUco marker detected. Calibration required')

  c = corners_list[0].reshape((4, 2))
  lado1 = np.linalg.norm(c[0] - c[1])
  lado2 = np.linalg.norm(c[1] - c[2])
  lado3 = np.linalg.norm(c[2] - c[3])
  lado4 = np.linalg.norm(c[3] - c[0])
  avg_pixels = np.mean([lado1, lado2, lado3, lado4])
  pixels_per_cm = avg_pixels / float(aruco_real_size_cm)

  exclude_mask = np.zeros_like(gray)
  for corner_set in corners_list:
    center = np.mean(corner_set, axis=0)
    expanded_pts = corner_set + (corner_set - center) * 0.15
    cv2.fillPoly(exclude_mask, [expanded_pts.astype(np.int32)], 255)

  methods = []
  adaptive_th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 21, 5)
  methods.append(('adaptive', adaptive_th))
  blur = cv2.GaussianBlur(gray, (7, 7), 0)
  _, otsu_th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
  methods.append(('otsu', otsu_th))
  edges = cv2.Canny(blur, 40, 130)
  kernel_canny = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
  canny_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_canny, iterations=2)
  methods.append(('canny', canny_closed))

  best_cnts = []
  for name, binary in methods:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (config['MORPH_KERNEL'], config['MORPH_KERNEL']))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.bitwise_and(cleaned, cv2.bitwise_not(exclude_mask))
    cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = []
    for c in cnts:
      if cv2.contourArea(c) < config['MIN_CONTOUR_AREA_PX']: continue
      rect = cv2.minAreaRect(c)
      _, (w, h), _ = rect
      if w == 0 or h == 0: continue
      aspect = max(w, h) / min(w, h) if min(w, h) > 0 else 0
      if aspect > config['MAX_ASPECT_RATIO']: continue
      hull = cv2.convexHull(c)
      hull_area = cv2.contourArea(hull)
      if hull_area > 0:
        solidity = cv2.contourArea(c) / hull_area
        if solidity < config['MIN_SOLIDITY']: continue
      valid.append(c)
    if len(valid) > len(best_cnts):
      best_cnts = valid

  results = []
  annotated = img.copy()

  for i, c in enumerate(best_cnts, start=1):
    peri = cv2.arcLength(c, True)
    area_px = cv2.contourArea(c)
    rect = cv2.minAreaRect(c)
    (center_px, (w_px, h_px), angle) = rect

    obj = {
      'index': i,
      'shape': 'Objeto',
      'area_px': float(area_px),
      'w_px': float(w_px),
      'h_px': float(h_px),
      'center_px': [int(center_px[0]), int(center_px[1])],
    }

    approx = None
    if peri > 0:
      try:
        approx = cv2.approxPolyDP(c, config['APPROX_EPS_FACTOR'] * peri, True)
      except Exception:
        approx = None

    n_vertices = len(approx) if approx is not None else 0

    if n_vertices == 3:
      shape = 'Triángulo'
    elif n_vertices == 4:
      aspect_ratio = max(w_px, h_px) / min(w_px, h_px) if min(w_px, h_px) > 0 else 0
      shape = 'Cuadrado' if aspect_ratio <= 1.2 else 'Rectángulo'
    elif n_vertices > 4:
      shape = f'Polígono ({n_vertices})'
    else:
      shape = 'Objeto'

    circularity = 0.0
    try:
      if peri > 0:
        circularity = 4 * math.pi * area_px / (peri * peri)
    except Exception:
      circularity = 0.0

    enclosing_area_ratio = 0.0
    try:
      (mx, my), r_px = cv2.minEnclosingCircle(c)
      enclosing_area = math.pi * (r_px ** 2)
      if enclosing_area > 0:
        enclosing_area_ratio = float(area_px) / float(enclosing_area)
    except Exception:
      enclosing_area_ratio = 0.0

    if peri > 0 and (circularity >= config.get('CIRCULARITY_THRESHOLD', 0.85) or enclosing_area_ratio >= 0.72):
      shape = 'Círculo'

    try:
      box = cv2.boxPoints(rect)
      box_area = cv2.contourArea(np.int0(box))
    except Exception:
      box = None
      box_area = 0

    box_fit_ratio = (area_px / box_area) if box_area > 0 else 0
    if shape != 'Círculo' and box_area > 0 and box_fit_ratio >= 0.70:
      shape = 'Rectángulo'
      if box is not None:
        try:
          approx = np.array(box, dtype=np.float32).reshape(-1, 1, 2)
          n_vertices = 4
        except Exception:
          pass

    try:
      hull = cv2.convexHull(c)
      hull_area = cv2.contourArea(hull)
      solidity = (area_px / hull_area) if hull_area > 0 else 1.0
    except Exception:
      hull_area = 0
      solidity = 1.0

    if shape != 'Círculo':
      if n_vertices >= 7 or (hull_area > 0 and solidity < max(0.45, config.get('MIN_SOLIDITY', 0.35) + 0.10)):
        shape = 'Objeto'

    obj['shape'] = shape

    w_cm = float(w_px) / pixels_per_cm
    h_cm = float(h_px) / pixels_per_cm
    obj['w_cm'] = w_cm
    obj['h_cm'] = h_cm

    perimeter_px = float(peri)
    perimeter_cm = float(perimeter_px) / pixels_per_cm if pixels_per_cm > 0 else 0.0
    obj['perimeter_px'] = perimeter_px
    obj['perimeter_cm'] = perimeter_cm

    area_cm2 = float(area_px) / (pixels_per_cm ** 2) if pixels_per_cm > 0 else 0.0
    obj['area_cm2'] = area_cm2

    sides_cm = []
    sides_px = []
    if approx is not None and n_vertices >= 3:
      pts = approx.reshape(-1, 2)
      for j in range(len(pts)):
        p1 = pts[j]
        p2 = pts[(j + 1) % len(pts)]
        d_px = float(np.linalg.norm(p1 - p2))
        sides_px.append(d_px)
        sides_cm.append(d_px / pixels_per_cm if pixels_per_cm > 0 else 0.0)
      obj['sides_cm'] = sides_cm
      obj['sides_px'] = sides_px
      obj['n_sides'] = len(pts)

    if shape.startswith('Triángulo') and 'sides_cm' in obj and len(obj['sides_cm']) == 3:
      a, b, c = obj['sides_cm']
      tol = 0.02 * max(a, b, c)
      if abs(a - b) <= tol and abs(b - c) <= tol:
        tri_type = 'Equilátero'
      elif abs(a - b) <= tol or abs(b - c) <= tol or abs(a - c) <= tol:
        tri_type = 'Isósceles'
      else:
        tri_type = 'Escaleno'
      obj['triangle_sides_cm'] = {'a_cm': a, 'b_cm': b, 'c_cm': c}
      obj['triangle_type'] = tri_type
      try:
        perimeter_sides = float(a + b + c)
        s = perimeter_sides / 2.0
        area_heron = math.sqrt(max(0.0, s * (s - a) * (s - b) * (s - c)))
        obj['perimeter_from_sides_cm'] = perimeter_sides
        obj['area_cm2'] = area_heron
        if area_heron > 0:
          diameter_circum = (a * b * c) / (2.0 * area_heron)
          obj['diameter_cm'] = diameter_circum
        else:
          obj['diameter_cm'] = None
      except Exception:
        pass

    if shape == 'Círculo':
      try:
        (_, r_px) = cv2.minEnclosingCircle(c)
        r_cm = float(r_px) / pixels_per_cm
        obj.update(
          {'radius_px': float(r_px), 'radius_cm': r_cm, 'diameter_cm': 2 * r_cm, 'perimeter_cm': 2 * math.pi * r_cm})
        cv2.circle(annotated, (int(center_px[0]), int(center_px[1])), int(r_px), (0, 255, 0), 3)
      except Exception:
        pass
    else:
      try:
        if box is None:
          box = cv2.boxPoints(rect)
        box_pts = box.reshape(-1, 2).tolist()
        obj['box_px'] = [[int(p[0]), int(p[1])] for p in box_pts]
        cv2.drawContours(annotated, [np.int0(box)], 0, (0, 255, 255), 2)
      except Exception:
        pass

    if 'diameter_cm' not in obj or obj.get('diameter_cm') is None:
      try:
        if shape.startswith('Rect') or shape.startswith('Cuadr'):
          diag = math.sqrt((w_cm ** 2) + (h_cm ** 2))
          obj['diameter_cm'] = float(diag)
        else:
          try:
            pts = c.reshape(-1, 2).astype(float)
            if pts.shape[0] >= 2:
              diffs = pts[:, None, :] - pts[None, :, :]
              dists = np.sqrt((diffs ** 2).sum(axis=2))
              max_px = float(dists.max())
              obj['diameter_cm'] = max_px / pixels_per_cm if pixels_per_cm > 0 else None
            else:
              obj['diameter_cm'] = None
          except Exception:
            obj['diameter_cm'] = None
      except Exception:
        obj['diameter_cm'] = None

    cx, cy = int(center_px[0]), int(center_px[1])
    cv2.putText(annotated, str(i), (cx - 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    try:
      contour_list = c.reshape(-1, 2).tolist()
      contour_list = [[int(p[0]), int(p[1])] for p in contour_list]
    except Exception:
      contour_list = []

    obj['contour'] = contour_list
    results.append(obj)

  timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
  out_dir = os.path.join(settings.MEDIA_ROOT, 'detection_results')
  ensure_dir(os.path.join(out_dir, 'x'))
  out_name = f"detection_{os.path.splitext(os.path.basename(image_path))[0]}_{timestamp}.png"
  out_path = os.path.join(out_dir, out_name)
  ensure_dir(out_path)
  cv2.imwrite(out_path, annotated)

  result_url = os.path.join(settings.MEDIA_URL, 'detection_results', out_name)

  for r in results:
    if r.get('area_cm2') is not None and r.get('area_cm2') != 0:
      continue
    if 'radius_cm' in r:
      r['area_cm2'] = math.pi * (r['radius_cm'] ** 2)
    else:
      r['area_cm2'] = float((r.get('w_cm', 0) * r.get('h_cm', 0)))

  return {
    'results': results,
    'pixels_per_cm': pixels_per_cm,
    'annotated_path': out_path,
    'annotated_url': result_url,
  }
