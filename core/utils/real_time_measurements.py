# """
# Real-time measurement calculations for shapes and objects
# """
# import cv2
# import numpy as np
# import math
# from typing import List, Tuple, Dict, Optional
# from .geometry import GeometryCalculator
#
#
# class RealTimeMeasurements:
#     """Real-time measurement calculator for various shapes and objects"""
#
#     def __init__(self):
#         self.geometry_calculator = GeometryCalculator()
#
#     def calculate_rectangle_measurements(self, points: List[Tuple[int, int]]) -> Dict:
#         """Calculate measurements for a rectangle"""
#         if len(points) < 2:
#             return {}
#
#         # Get bounding rectangle
#         x_coords = [p[0] for p in points]
#         y_coords = [p[1] for p in points]
#
#         min_x, max_x = min(x_coords), max(x_coords)
#         min_y, max_y = min(y_coords), max(y_coords)
#
#         width = max_x - min_x
#         height = max_y - min_y
#
#         # Calculate measurements
#         area = width * height
#         perimeter = 2 * (width + height)
#         diagonal = math.sqrt(width**2 + height**2)
#
#         return {
#             'type': 'rectangle',
#             'width': width,
#             'height': height,
#             'area': area,
#             'perimeter': perimeter,
#             'diagonal': diagonal,
#             'aspect_ratio': width / height if height > 0 else 0,
#             'center': ((min_x + max_x) / 2, (min_y + max_y) / 2)
#         }
#
#     def calculate_circle_measurements(self, center: Tuple[int, int], radius: float) -> Dict:
#         """Calculate measurements for a circle"""
#         area = math.pi * radius**2
#         perimeter = 2 * math.pi * radius
#         diameter = 2 * radius
#
#         return {
#             'type': 'circle',
#             'radius': radius,
#             'diameter': diameter,
#             'area': area,
#             'perimeter': perimeter,
#             'center': center
#         }
#
#     def calculate_triangle_measurements(self, points: List[Tuple[int, int]]) -> Dict:
#         """Calculate measurements for a triangle"""
#         if len(points) != 3:
#             return {}
#
#         p1, p2, p3 = points
#
#         # Calculate side lengths
#         side1 = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
#         side2 = math.sqrt((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2)
#         side3 = math.sqrt((p1[0] - p3[0])**2 + (p1[1] - p3[1])**2)
#
#         # Calculate perimeter
#         perimeter = side1 + side2 + side3
#
#         # Calculate area using Heron's formula
#         s = perimeter / 2
#         area = math.sqrt(s * (s - side1) * (s - side2) * (s - side3))
#
#         # Calculate center (centroid)
#         center_x = (p1[0] + p2[0] + p3[0]) / 3
#         center_y = (p1[1] + p2[1] + p3[1]) / 3
#
#         return {
#             'type': 'triangle',
#             'side1': side1,
#             'side2': side2,
#             'side3': side3,
#             'area': area,
#             'perimeter': perimeter,
#             'center': (center_x, center_y)
#         }
#
#     def calculate_polygon_measurements(self, points: List[Tuple[int, int]]) -> Dict:
#         """Calculate measurements for a polygon"""
#         if len(points) < 3:
#             return {}
#
#         # Calculate area using shoelace formula
#         area = 0
#         perimeter = 0
#
#         n = len(points)
#         for i in range(n):
#             j = (i + 1) % n
#             area += points[i][0] * points[j][1]
#             area -= points[j][0] * points[i][1]
#
#             # Calculate perimeter
#             perimeter += math.sqrt(
#                 (points[j][0] - points[i][0])**2 +
#                 (points[j][1] - points[i][1])**2
#             )
#
#         area = abs(area) / 2
#
#         # Calculate center
#         center_x = sum(p[0] for p in points) / len(points)
#         center_y = sum(p[1] for p in points) / len(points)
#
#         return {
#             'type': 'polygon',
#             'vertices': len(points),
#             'area': area,
#             'perimeter': perimeter,
#             'center': (center_x, center_y)
#         }
#
#     def calculate_line_measurements(self, points: List[Tuple[int, int]]) -> Dict:
#         """Calculate measurements for a line"""
#         if len(points) != 2:
#             return {}
#
#         p1, p2 = points
#
#         # Calculate distance
#         distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
#
#         # Calculate angle
#         angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
#
#         # Calculate midpoint
#         midpoint = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
#
#         return {
#             'type': 'line',
#             'distance': distance,
#             'angle': angle,
#             'midpoint': midpoint,
#             'start_point': p1,
#             'end_point': p2
#         }
#
#     def detect_shape_type(self, points: List[Tuple[int, int]]) -> str:
#         """Detect the type of shape based on points"""
#         if len(points) < 2:
#             return 'unknown'
#         elif len(points) == 2:
#             return 'line'
#         elif len(points) == 3:
#             return 'triangle'
#         elif len(points) == 4:
#             # Check if it's a rectangle
#             return 'rectangle'
#         else:
#             return 'polygon'
#
#     def calculate_measurements(self, points: List[Tuple[int, int]], shape_type: str = None) -> Dict:
#         """Calculate measurements for any shape type"""
#         if not points:
#             return {}
#
#         if shape_type is None:
#             shape_type = self.detect_shape_type(points)
#
#         if shape_type == 'line':
#             return self.calculate_line_measurements(points)
#         elif shape_type == 'triangle':
#             return self.calculate_triangle_measurements(points)
#         elif shape_type == 'rectangle':
#             return self.calculate_rectangle_measurements(points)
#         elif shape_type == 'polygon':
#             return self.calculate_polygon_measurements(points)
#         else:
#             return {}
#
#     def convert_to_real_units(self, measurements: Dict, pixels_per_cm: float) -> Dict:
#         """Convert pixel measurements to real units (cm)"""
#         if not measurements or not pixels_per_cm:
#             return measurements
#
#         real_measurements = measurements.copy()
#
#         # Convert linear measurements
#         linear_fields = ['width', 'height', 'distance', 'radius', 'diameter',
#                         'side1', 'side2', 'side3', 'perimeter', 'diagonal']
#
#         for field in linear_fields:
#             if field in real_measurements:
#                 real_measurements[f'{field}_cm'] = real_measurements[field] / pixels_per_cm
#
#         # Convert area measurements
#         if 'area' in real_measurements:
#             real_measurements['area_cm2'] = real_measurements['area'] / (pixels_per_cm ** 2)
#
#         return real_measurements
#
#     def get_measurement_summary(self, measurements: Dict) -> str:
#         """Get a human-readable summary of measurements"""
#         if not measurements:
#             return "No measurements available"
#
#         summary_parts = []
#
#         if measurements.get('type') == 'line':
#             distance = measurements.get('distance', 0)
#             angle = measurements.get('angle', 0)
#             summary_parts.append(f"Distancia: {distance:.1f}px")
#             summary_parts.append(f"Ángulo: {angle:.1f}°")
#
#         elif measurements.get('type') == 'rectangle':
#             width = measurements.get('width', 0)
#             height = measurements.get('height', 0)
#             area = measurements.get('area', 0)
#             summary_parts.append(f"Ancho: {width:.1f}px")
#             summary_parts.append(f"Alto: {height:.1f}px")
#             summary_parts.append(f"Área: {area:.1f}px²")
#
#         elif measurements.get('type') == 'circle':
#             radius = measurements.get('radius', 0)
#             area = measurements.get('area', 0)
#             summary_parts.append(f"Radio: {radius:.1f}px")
#             summary_parts.append(f"Área: {area:.1f}px²")
#
#         elif measurements.get('type') == 'triangle':
#             area = measurements.get('area', 0)
#             perimeter = measurements.get('perimeter', 0)
#             summary_parts.append(f"Área: {area:.1f}px²")
#             summary_parts.append(f"Perímetro: {perimeter:.1f}px")
#
#         elif measurements.get('type') == 'polygon':
#             area = measurements.get('area', 0)
#             perimeter = measurements.get('perimeter', 0)
#             vertices = measurements.get('vertices', 0)
#             summary_parts.append(f"Vértices: {vertices}")
#             summary_parts.append(f"Área: {area:.1f}px²")
#             summary_parts.append(f"Perímetro: {perimeter:.1f}px")
#
#         return " | ".join(summary_parts)
#
#
# class GestureMeasurementTracker:
#     """Track measurements from gesture inputs"""
#
#     def __init__(self):
#         self.measurement_calculator = RealTimeMeasurements()
#         self.current_points = []
#         self.current_shape_type = None
#         self.pixels_per_cm = None
#
#     def add_point(self, point: Tuple[int, int]):
#         """Add a measurement point"""
#         self.current_points.append(point)
#
#         # Auto-detect shape type
#         self.current_shape_type = self.measurement_calculator.detect_shape_type(self.current_points)
#
#     def clear_points(self):
#         """Clear all measurement points"""
#         self.current_points = []
#         self.current_shape_type = None
#
#     def get_current_measurements(self) -> Dict:
#         """Get current measurements based on points"""
#         if not self.current_points:
#             return {}
#
#         measurements = self.measurement_calculator.calculate_measurements(
#             self.current_points,
#             self.current_shape_type
#         )
#
#         # Convert to real units if calibrated
#         if self.pixels_per_cm:
#             measurements = self.measurement_calculator.convert_to_real_units(
#                 measurements,
#                 self.pixels_per_cm
#             )
#
#         return measurements
#
#     def get_measurement_summary(self) -> str:
#         """Get measurement summary"""
#         measurements = self.get_current_measurements()
#         return self.measurement_calculator.get_measurement_summary(measurements)
#
#     def set_calibration(self, pixels_per_cm: float):
#         """Set calibration for real unit conversion"""
#         self.pixels_per_cm = pixels_per_cm
#
#     def is_ready_for_measurement(self) -> bool:
#         """Check if enough points for measurement"""
#         return len(self.current_points) >= 2
#
#     def get_shape_type(self) -> str:
#         """Get current shape type"""
#         return self.current_shape_type or 'unknown'
