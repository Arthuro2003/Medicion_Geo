from datetime import timedelta
from typing import Any, Union

from django import template

register = template.Library()

SECONDS_PER_HOUR = 3600
SECONDS_PER_MINUTE = 60
DEFAULT_DECIMAL_PLACES = 2


def safe_float_conversion(value: Any) -> Union[float, Any]:
  try:
    return float(value)
  except (ValueError, TypeError):
    return value


@register.filter
def two_decimals(value: Any) -> str:
  try:
    return f"{float(value):.{DEFAULT_DECIMAL_PLACES}f}"
  except (ValueError, TypeError):
    return value


@register.filter
def human_seconds(value: Any) -> str:
  seconds = safe_float_conversion(value)
  if isinstance(seconds, float):
    if seconds < SECONDS_PER_MINUTE:
      return f"{seconds:.{DEFAULT_DECIMAL_PLACES}f} s"

    hours = int(seconds // SECONDS_PER_HOUR)
    minutes = int((seconds % SECONDS_PER_HOUR) // SECONDS_PER_MINUTE)
    remaining_seconds = int(seconds % SECONDS_PER_MINUTE)

    if hours > 0:
      return f"{hours}:{minutes:02d}:{remaining_seconds:02d}"
    return f"{minutes}:{remaining_seconds:02d}"

  return value


@register.filter
def timedelta_seconds(value: timedelta) -> Union[float, Any]:
  try:
    return float(value.total_seconds())
  except (AttributeError, ValueError, TypeError):
    return value


@register.filter
def seconds_since(value: Any, base_datetime: Any) -> Union[float, Any]:
  timestamp = safe_float_conversion(value)
  if isinstance(timestamp, float):
    try:
      if hasattr(base_datetime, 'timestamp'):
        base_ts = float(base_datetime.timestamp())
        difference = timestamp - base_ts
        return float(max(difference, 0.0))
    except (AttributeError, ValueError, TypeError):
      pass

  return value
