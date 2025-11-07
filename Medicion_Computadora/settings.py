"""
Django settings for Medicion_Computadora project using python-dotenv

This file loads environment variables from a .env file with python-dotenv
and otherwise reads values from the process environment (os.environ).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(dotenv_path=str(BASE_DIR / '.env'))


def _bool_env(key: str, default: bool = False) -> bool:
  val = os.getenv(key)
  if val is None:
    return default
  return val.lower() in ('1', 'true', 'yes', 'on')


def _int_env(key: str, default: int = 0) -> int:
  val = os.getenv(key)
  try:
    return int(val) if val is not None else default
  except (ValueError, TypeError):
    return default


SECRET_KEY = os.getenv('SECRET_KEY') or 'django-insecure-vko%@t2^v!glgxph5k$vrsvhqb#u@8qh(ej5jslk%9w9#9_#kb'

DEBUG = _bool_env('DEBUG', True)

ALLOWED_HOSTS = [h.strip() for h in os.getenv('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',') if h.strip()]

INSTALLED_APPS = [
  'django.contrib.admin',
  'django.contrib.auth',
  'django.contrib.contenttypes',
  'django.contrib.sessions',
  'django.contrib.messages',
  'django.contrib.staticfiles',
  'crispy_forms',
  'crispy_bootstrap5',
  'core.apps.CoreConfig',
]

if DEBUG:
  try:
    import debug_toolbar  # noqa: F401

    INSTALLED_APPS += ['debug_toolbar']
  except Exception:
    pass

MIDDLEWARE = [
  'django.middleware.security.SecurityMiddleware',
  'django.contrib.sessions.middleware.SessionMiddleware',
  'django.middleware.common.CommonMiddleware',
  'django.middleware.csrf.CsrfViewMiddleware',
  'django.contrib.auth.middleware.AuthenticationMiddleware',
  'django.contrib.messages.middleware.MessageMiddleware',
  'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

if DEBUG and 'debug_toolbar' in INSTALLED_APPS:
  MIDDLEWARE = ['debug_toolbar.middleware.DebugToolbarMiddleware'] + MIDDLEWARE

ROOT_URLCONF = 'Medicion_Computadora.urls'

TEMPLATES = [
  {
    'BACKEND': 'django.template.backends.django.DjangoTemplates',
    'DIRS': [BASE_DIR / 'templates'],
    'APP_DIRS': True,
    'OPTIONS': {
      'context_processors': [
        'django.template.context_processors.request',
        'django.contrib.auth.context_processors.auth',
        'django.contrib.messages.context_processors.messages',
        'core.context_processors.user_preferences',
      ],
    },
  },
]

WSGI_APPLICATION = 'Medicion_Computadora.wsgi.application'

if os.getenv('DB_NAME') and os.getenv('DB_USER') and os.getenv('DB_PASSWORD'):
  DATABASES = {
    'default': {
      'ENGINE': 'django.db.backends.postgresql',
      'NAME': os.getenv('DB_NAME'),
      'USER': os.getenv('DB_USER'),
      'PASSWORD': os.getenv('DB_PASSWORD'),
      'HOST': os.getenv('DB_HOST', 'localhost'),
      'PORT': os.getenv('DB_PORT', '5432'),
    }
  }
else:
  DATABASES = {
    'default': {
      'ENGINE': 'django.db.backends.sqlite3',
      'NAME': str(BASE_DIR / 'db.sqlite3'),
    }
  }

AUTH_PASSWORD_VALIDATORS = [
  {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
  {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
  {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
  {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

LANGUAGE_CODE = os.getenv('LANGUAGE_CODE', 'es-ec')
TIME_ZONE = os.getenv('TIME_ZONE', 'America/Guayaquil')
USE_I18N = True
USE_TZ = True

STATIC_URL = os.getenv('STATIC_URL', '/static/')
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_DIRS = [BASE_DIR / 'static']

MEDIA_URL = os.getenv('MEDIA_URL', '/media/')
MEDIA_ROOT = BASE_DIR / 'media'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

LOGIN_REDIRECT_URL = os.getenv('LOGIN_REDIRECT_URL', '')
LOGOUT_REDIRECT_URL = os.getenv('LOGOUT_REDIRECT_URL', '')

FILE_UPLOAD_MAX_MEMORY_SIZE = _int_env('FILE_UPLOAD_MAX_MEMORY_SIZE', 10 * 1024 * 1024)
DATA_UPLOAD_MAX_MEMORY_SIZE = _int_env('DATA_UPLOAD_MAX_MEMORY_SIZE', 10 * 1024 * 1024)

MEASUREMENT_SETTINGS = {
  'MAX_IMAGE_SIZE': (4000, 4000),
  'THUMBNAIL_SIZE': (300, 200),
  'ALLOWED_FORMATS': ['JPEG', 'PNG', 'BMP', 'TIFF'],
  'DETECTION_CONFIDENCE': float(os.getenv('DETECTION_CONFIDENCE', '0.7')),
  'MIN_CONTOUR_AREA': int(os.getenv('MIN_CONTOUR_AREA', '100')),
}

REPORT_SETTINGS = {
  'OUTPUT_DIR': MEDIA_ROOT / 'reports',
  'PDF_PAGE_SIZE': os.getenv('PDF_PAGE_SIZE', 'A4'),
  'EXCEL_ENGINE': os.getenv('EXCEL_ENGINE', 'openpyxl'),
  'MAX_IMAGES_IN_PDF': _int_env('MAX_IMAGES_IN_PDF', 20),
  'CLEANUP_OLD_REPORTS_DAYS': _int_env('CLEANUP_OLD_REPORTS_DAYS', 30),
}

os.makedirs(REPORT_SETTINGS['OUTPUT_DIR'], exist_ok=True)

CACHES = {
  'default': {
    'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
    'LOCATION': 'measurement-cache',
    'TIMEOUT': _int_env('CACHE_TIMEOUT', 300),
    'OPTIONS': {'MAX_ENTRIES': _int_env('CACHE_MAX_ENTRIES', 1000)},
  }
}

LOGGING = {
  'version': 1,
  'disable_existing_loggers': False,
  'formatters': {
    'verbose': {'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}', 'style': '{'},
    'simple': {'format': '{levelname} {message}', 'style': '{'},
  },
  'handlers': {
    'file': {
      'level': 'INFO',
      'class': 'logging.FileHandler',
      'filename': str(BASE_DIR / 'logs' / 'django.log'),
      'formatter': 'verbose',
    },
    'console': {'level': 'DEBUG', 'class': 'logging.StreamHandler', 'formatter': 'simple'},
  },
  'root': {'handlers': ['console'], 'level': 'INFO'},
  'loggers': {
    'django': {'handlers': ['file', 'console'], 'level': 'INFO', 'propagate': False},
    'core': {'handlers': ['file', 'console'], 'level': 'DEBUG', 'propagate': False},
  },
}

os.makedirs(BASE_DIR / 'logs', exist_ok=True)

if not DEBUG:
  SECURE_SSL_REDIRECT = _bool_env('SECURE_SSL_REDIRECT', True)
  SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
  SECURE_BROWSER_XSS_FILTER = True
  SECURE_CONTENT_TYPE_NOSNIFF = True
  X_FRAME_OPTIONS = 'DENY'
  SECURE_HSTS_SECONDS = _int_env('SECURE_HSTS_SECONDS', 31536000)
  SECURE_HSTS_INCLUDE_SUBDOMAINS = _bool_env('SECURE_HSTS_INCLUDE_SUBDOMAINS', True)
  SECURE_HSTS_PRELOAD = _bool_env('SECURE_HSTS_PRELOAD', True)
  SESSION_COOKIE_SECURE = True
  CSRF_COOKIE_SECURE = True
  ALLOWED_HOSTS = [h.strip() for h in os.getenv('ALLOWED_HOSTS', ','.join(ALLOWED_HOSTS)).split(',') if h.strip()]

if not DEBUG:
  EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
  EMAIL_HOST = os.getenv('EMAIL_HOST', 'localhost')
  EMAIL_PORT = _int_env('EMAIL_PORT', 587)
  EMAIL_USE_TLS = _bool_env('EMAIL_USE_TLS', True)
  EMAIL_HOST_USER = os.getenv('EMAIL_HOST_USER', '')
  EMAIL_HOST_PASSWORD = os.getenv('EMAIL_HOST_PASSWORD', '')
  DEFAULT_FROM_EMAIL = os.getenv('DEFAULT_FROM_EMAIL', 'noreply@mediciongeo.com')
else:
  EMAIL_BACKEND = os.getenv('EMAIL_BACKEND', 'django.core.mail.backends.console.EmailBackend')
  EMAIL_HOST = os.getenv('EMAIL_HOST', 'localhost')
  EMAIL_PORT = _int_env('EMAIL_PORT', 587)
  EMAIL_USE_TLS = _bool_env('EMAIL_USE_TLS', True)
  EMAIL_HOST_USER = os.getenv('EMAIL_HOST_USER', '')
  EMAIL_HOST_PASSWORD = os.getenv('EMAIL_HOST_PASSWORD', '')
  DEFAULT_FROM_EMAIL = os.getenv('DEFAULT_FROM_EMAIL', 'noreply@mediciongeo.com')

CRISPY_ALLOWED_TEMPLATE_PACKS = "bootstrap5"
CRISPY_TEMPLATE_PACK = "bootstrap5"

REST_FRAMEWORK = {
  'DEFAULT_AUTHENTICATION_CLASSES': ['rest_framework.authentication.SessionAuthentication'],
  'DEFAULT_PERMISSION_CLASSES': ['rest_framework.permissions.IsAuthenticated'],
  'DEFAULT_RENDERER_CLASSES': ['rest_framework.renderers.JSONRenderer'],
  'PAGE_SIZE': _int_env('API_PAGE_SIZE', 20),
}
