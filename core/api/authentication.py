import secrets
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import jwt
from django.conf import settings
from django.contrib.auth.models import User
from django.db import models
from django.http import JsonResponse
from rest_framework import status, permissions
from rest_framework.authentication import BaseAuthentication
from rest_framework.permissions import BasePermission
from rest_framework.response import Response
from rest_framework.views import APIView


class OAuth2Application(models.Model):
  GRANT_TYPES = [
    ('authorization_code', 'Código de autorización'),
    ('client_credentials', 'Credenciales de cliente'),
    ('refresh_token', 'Token de actualización'),
  ]

  CLIENT_TYPES = [
    ('confidential', 'Confidencial'),
    ('public', 'Público'),
  ]

  name = models.CharField(max_length=255, verbose_name="Nombre de la aplicación")
  client_id = models.CharField(max_length=100, unique=True, verbose_name="Cliente ID")
  client_secret = models.CharField(max_length=255, verbose_name="Cliente Secreto")
  client_type = models.CharField(max_length=20, choices=CLIENT_TYPES, default='confidential')
  authorization_grant_type = models.CharField(max_length=32, choices=GRANT_TYPES, default='authorization_code')

  redirect_uris = models.TextField(help_text="Introduce varias URIs separadas por saltos de línea")
  homepage_url = models.URLField(blank=True, verbose_name="URL de la página principal")

  scopes = models.TextField(default="read", help_text="Alcances separados por espacios")

  user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='oauth_applications')
  created_at = models.DateTimeField(auto_now_add=True)
  updated_at = models.DateTimeField(auto_now=True)
  is_active = models.BooleanField(default=True)

  class Meta:
    verbose_name = "Aplicación OAuth2"
    verbose_name_plural = "Aplicaciones OAuth2"

  def __str__(self):
    return self.name

  def save(self, *args, **kwargs):
    if not self.client_id:
      self.client_id = self._generate_client_id()
    if not self.client_secret:
      self.client_secret = self._generate_client_secret()
    super().save(*args, **kwargs)

  def _generate_client_id(self):
    return f"medicion_{secrets.token_urlsafe(16)}"

  def _generate_client_secret(self):
    return secrets.token_urlsafe(32)

  def get_redirect_uris(self):
    return [uri.strip() for uri in self.redirect_uris.split('\n') if uri.strip()]

  def get_scopes_list(self):
    return [scope.strip() for scope in self.scopes.split(' ') if scope.strip()]


class ExpirationMixin:
  def is_expired(self):
    return datetime.now() > self.expires_at

  def is_valid(self):
    return not getattr(self, 'revoked', False) and not getattr(self, 'used', False) and not self.is_expired()


class OAuth2AuthorizationCode(models.Model, ExpirationMixin):
  application = models.ForeignKey(OAuth2Application, on_delete=models.CASCADE)
  user = models.ForeignKey(User, on_delete=models.CASCADE)
  code = models.CharField(max_length=255, unique=True)
  redirect_uri = models.URLField()
  scopes = models.TextField()
  code_challenge = models.CharField(max_length=255, blank=True)
  code_challenge_method = models.CharField(max_length=10, blank=True)
  created_at = models.DateTimeField(auto_now_add=True)
  expires_at = models.DateTimeField()
  used = models.BooleanField(default=False)

  class Meta:
    verbose_name = "Código de autorización OAuth2"
    verbose_name_plural = "Códigos de autorización OAuth2"

  def save(self, *args, **kwargs):
    if not self.code:
      self.code = secrets.token_urlsafe(32)
    if not self.expires_at:
      self.expires_at = datetime.now() + timedelta(minutes=10)
    super().save(*args, **kwargs)


class TokenMixin:
  def get_scopes_list(self):
    return [scope.strip() for scope in self.scopes.split(' ') if scope.strip()]


class OAuth2AccessToken(models.Model, ExpirationMixin, TokenMixin):
  TOKEN_TYPES = [
    ('Bearer', 'Bearer'),
  ]

  application = models.ForeignKey(OAuth2Application, on_delete=models.CASCADE)
  user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
  token = models.TextField(unique=True)
  token_type = models.CharField(max_length=20, choices=TOKEN_TYPES, default='Bearer')
  scopes = models.TextField()
  expires_at = models.DateTimeField()
  created_at = models.DateTimeField(auto_now_add=True)
  revoked = models.BooleanField(default=False)
  refresh_token = models.CharField(max_length=255, blank=True, unique=True)

  class Meta:
    verbose_name = "Token de acceso OAuth2"
    verbose_name_plural = "Tokens de acceso OAuth2"

  def save(self, *args, **kwargs):
    if not self.token:
      self.token = self._generate_jwt_token()
    if not self.refresh_token and 'refresh_token' in self.get_scopes_list():
      self.refresh_token = secrets.token_urlsafe(32)
    if not self.expires_at:
      self.expires_at = datetime.now() + timedelta(hours=1)
    super().save(*args, **kwargs)

  def _generate_jwt_token(self):
    payload = {
      'client_id': self.application.client_id,
      'user_id': self.user.id if self.user else None,
      'scopes': self.scopes,
      'iat': datetime.now(timezone.utc),
      'exp': self.expires_at or (datetime.now(timezone.utc) + timedelta(hours=1)),
      'iss': 'medicion-geo-api',
      'aud': 'medicion-geo-client'
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm='HS256')


class OAuth2RefreshToken(models.Model, ExpirationMixin):
  application = models.ForeignKey(OAuth2Application, on_delete=models.CASCADE)
  user = models.ForeignKey(User, on_delete=models.CASCADE)
  token = models.CharField(max_length=255, unique=True)
  access_token = models.ForeignKey(OAuth2AccessToken, on_delete=models.CASCADE, related_name='refresh_tokens')
  created_at = models.DateTimeField(auto_now_add=True)
  expires_at = models.DateTimeField()
  revoked = models.BooleanField(default=False)

  class Meta:
    verbose_name = "Token de actualización OAuth2"
    verbose_name_plural = "Tokens de actualización OAuth2"

  def save(self, *args, **kwargs):
    if not self.token:
      self.token = secrets.token_urlsafe(32)
    if not self.expires_at:
      self.expires_at = datetime.now() + timedelta(days=30)
    super().save(*args, **kwargs)


class BearerAuthenticationMixin:
  def get_bearer_token(self, request):
    auth_header = request.META.get('HTTP_AUTHORIZATION')
    if not auth_header or not auth_header.startswith('Bearer '):
      return None
    return auth_header.split(' ')[1]


class JWTAuthentication(BaseAuthentication, BearerAuthenticationMixin):
  def authenticate(self, request):
    token = self.get_bearer_token(request)
    if not token:
      return None

    try:
      payload = jwt.decode(token, settings.SECRET_KEY, algorithms=['HS256'])
      user_id = payload.get('user_id')
      if user_id:
        user = User.objects.get(id=user_id)
        return user, token
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError, User.DoesNotExist):
      pass
    return None


class OAuth2Authentication(BaseAuthentication, BearerAuthenticationMixin):
  def authenticate(self, request):
    token = self.get_bearer_token(request)
    if not token:
      return None

    try:
      access_token = OAuth2AccessToken.objects.get(token=token)
      if access_token.is_valid():
        return access_token.user, access_token
    except OAuth2AccessToken.DoesNotExist:
      pass
    return None


class IsOwnerOrReadOnly(BasePermission):

  def has_object_permission(self, request, view, obj):
    if request.method in permissions.SAFE_METHODS:
      return True

    return obj.user == request.user


class HasScopePermission(BasePermission):

  def __init__(self, required_scopes):
    self.required_scopes = required_scopes

  def has_permission(self, request, view):
    if not hasattr(request, 'auth') or not request.auth:
      return False

    if isinstance(request.auth, OAuth2AccessToken):
      token_scopes = request.auth.get_scopes_list()
      return all(scope in token_scopes for scope in self.required_scopes)

    return False


class OAuth2AuthorizationView(APIView):

  def get(self, request):
    client_id = request.GET.get('client_id')
    redirect_uri = request.GET.get('redirect_uri')
    response_type = request.GET.get('response_type')
    scope = request.GET.get('scope', 'read')
    state = request.GET.get('state')

    if not all([client_id, redirect_uri, response_type]):
      return Response({
        'error': 'solicitud_invalida',
        'error_description': 'Faltan parámetros requeridos'
      }, status=status.HTTP_400_BAD_REQUEST)

    try:
      application = OAuth2Application.objects.get(
        client_id=client_id,
        is_active=True
      )
    except OAuth2Application.DoesNotExist:
      return Response({
        'error': 'invalid_client',
        'error_description': 'Client_id inválido'
      }, status=status.HTTP_400_BAD_REQUEST)

    if redirect_uri not in application.get_redirect_uris():
      return Response({
        'error': 'invalid_request',
        'error_description': 'Redirect_uri inválida'
      }, status=status.HTTP_400_BAD_REQUEST)

    if response_type != 'code':
      return Response({
        'error': 'unsupported_response_type',
        'error_description': 'Solo se soporta el flujo authorization_code'
      }, status=status.HTTP_400_BAD_REQUEST)

    request.session['oauth2_auth'] = {
      'client_id': client_id,
      'redirect_uri': redirect_uri,
      'scope': scope,
      'state': state
    }

    return Response({
      'authorization_url': f"{settings.FRONTEND_URL}/oauth/authorize?client_id={client_id}&redirect_uri={redirect_uri}&response_type={response_type}&scope={scope}&state={state}"
    })


class OAuth2TokenView(APIView):

  def post(self, request):
    grant_type = request.data.get('grant_type')

    if grant_type == 'authorization_code':
      return self._handle_authorization_code(request)
    elif grant_type == 'client_credentials':
      return self._handle_client_credentials(request)
    elif grant_type == 'refresh_token':
      return self._handle_refresh_token(request)
    else:
      return Response({
        'error': 'unsupported_grant_type',
        'error_description': 'Tipo de grant no soportado'
      }, status=status.HTTP_400_BAD_REQUEST)

  def _handle_authorization_code(self, request):
    code = request.data.get('code')
    redirect_uri = request.data.get('redirect_uri')
    client_id = request.data.get('client_id')
    client_secret = request.data.get('client_secret')

    try:
      auth_code = OAuth2AuthorizationCode.objects.get(
        code=code,
        used=False
      )
    except OAuth2AuthorizationCode.DoesNotExist:
      return Response({
        'error': 'invalid_grant',
        'error_description': 'Código de autorización inválido'
      }, status=status.HTTP_400_BAD_REQUEST)

    if auth_code.is_expired():
      return Response({
        'error': 'invalid_grant',
        'error_description': 'Código de autorización expirado'
      }, status=status.HTTP_400_BAD_REQUEST)

    if auth_code.redirect_uri != redirect_uri:
      return Response({
        'error': 'invalid_grant',
        'error_description': 'Redirect_uri inválida'
      }, status=status.HTTP_400_BAD_REQUEST)

    if auth_code.application.client_id != client_id:
      return Response({
        'error': 'invalid_client',
        'error_description': 'Client_id inválido'
      }, status=status.HTTP_400_BAD_REQUEST)

    if auth_code.application.client_secret != client_secret:
      return Response({
        'error': 'invalid_client',
        'error_description': 'Client_secret inválido'
      }, status=status.HTTP_400_BAD_REQUEST)

    auth_code.used = True
    auth_code.save()

    access_token = OAuth2AccessToken.objects.create(
      application=auth_code.application,
      user=auth_code.user,
      scopes=auth_code.scopes
    )

    refresh_token = None
    if 'refresh_token' in auth_code.scopes:
      refresh_token = OAuth2RefreshToken.objects.create(
        application=auth_code.application,
        user=auth_code.user,
        access_token=access_token
      )

    response_data = {
      'access_token': access_token.token,
      'token_type': 'Bearer',
      'expires_in': 3600,
      'scope': access_token.scopes
    }

    if refresh_token:
      response_data['refresh_token'] = refresh_token.token

    return Response(response_data)

  def _handle_client_credentials(self, request):
    client_id = request.data.get('client_id')
    client_secret = request.data.get('client_secret')
    scope = request.data.get('scope', 'read')

    try:
      application = OAuth2Application.objects.get(
        client_id=client_id,
        client_secret=client_secret,
        is_active=True
      )
    except OAuth2Application.DoesNotExist:
      return Response({
        'error': 'invalid_client',
        'error_description': 'Credenciales de cliente inválidas'
      }, status=status.HTTP_400_BAD_REQUEST)

    access_token = OAuth2AccessToken.objects.create(
      application=application,
      user=None,
      scopes=scope
    )

    return Response({
      'access_token': access_token.token,
      'token_type': 'Bearer',
      'expires_in': 3600,
      'scope': access_token.scopes
    })

  def _handle_refresh_token(self, request):
    refresh_token = request.data.get('refresh_token')
    client_id = request.data.get('client_id')
    client_secret = request.data.get('client_secret')

    try:
      token = OAuth2RefreshToken.objects.get(
        token=refresh_token,
        revoked=False
      )
    except OAuth2RefreshToken.DoesNotExist:
      return Response({
        'error': 'invalid_grant',
        'error_description': 'Token de actualización inválido'
      }, status=status.HTTP_400_BAD_REQUEST)

    if token.is_expired():
      return Response({
        'error': 'invalid_grant',
        'error_description': 'Token de actualización expirado'
      }, status=status.HTTP_400_BAD_REQUEST)

    if token.application.client_id != client_id:
      return Response({
        'error': 'invalid_client',
        'error_description': 'Client_id inválido'
      }, status=status.HTTP_400_BAD_REQUEST)

    if token.application.client_secret != client_secret:
      return Response({
        'error': 'invalid_client',
        'error_description': 'Client_secret inválido'
      }, status=status.HTTP_400_BAD_REQUEST)

    token.access_token.revoked = True
    token.access_token.save()

    new_access_token = OAuth2AccessToken.objects.create(
      application=token.application,
      user=token.user,
      scopes=token.access_token.scopes
    )

    new_refresh_token = OAuth2RefreshToken.objects.create(
      application=token.application,
      user=token.user,
      access_token=new_access_token
    )

    token.revoked = True
    token.save()

    return Response({
      'access_token': new_access_token.token,
      'token_type': 'Bearer',
      'expires_in': 3600,
      'scope': new_access_token.scopes,
      'refresh_token': new_refresh_token.token
    })


class JWTTokenManager:

  @staticmethod
  def generate_token(user: User, additional_payload: Dict = None) -> str:
    payload = {
      'user_id': user.id,
      'username': user.username,
      'email': user.email,
      'iat': datetime.now(timezone.utc),
      'exp': datetime.now(timezone.utc) + timedelta(hours=24),
      'iss': 'medicion-geo-api',
      'aud': 'medicion-geo-client'
    }

    if additional_payload:
      payload.update(additional_payload)

    return jwt.encode(payload, settings.SECRET_KEY, algorithm='HS256')

  @staticmethod
  def validate_token(token: str) -> Optional[Dict]:
    try:
      payload = jwt.decode(token, settings.SECRET_KEY, algorithms=['HS256'])
      return payload
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
      return None

  @staticmethod
  def refresh_token(token: str) -> Optional[str]:
    payload = JWTTokenManager.validate_token(token)
    if payload and 'user_id' in payload:
      try:
        user = User.objects.get(id=payload['user_id'])
        return JWTTokenManager.generate_token(user)
      except User.DoesNotExist:
        pass
    return None


class LoginView(APIView):

  def post(self, request):
    username = request.data.get('username')
    password = request.data.get('password')

    if not username or not password:
      return Response({
        'error': 'missing_credentials',
        'error_description': 'Nombre de usuario y contraseña son requeridos'
      }, status=status.HTTP_400_BAD_REQUEST)

    try:
      user = User.objects.get(username=username)
      if user.check_password(password):
        token = JWTTokenManager.generate_token(user)
        return Response({
          'access_token': token,
          'token_type': 'Bearer',
          'expires_in': 86400,
          'user': {
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'first_name': user.first_name,
            'last_name': user.last_name
          }
        })
      else:
        return Response({
          'error': 'invalid_credentials',
          'error_description': 'Nombre de usuario o contraseña inválidos'
        }, status=status.HTTP_401_UNAUTHORIZED)
    except User.DoesNotExist:
      return Response({
        'error': 'invalid_credentials',
        'error_description': 'Nombre de usuario o contraseña inválidos'
      }, status=status.HTTP_401_UNAUTHORIZED)


class LogoutView(APIView):
  authentication_classes = [JWTAuthentication]
  permission_classes = [permissions.IsAuthenticated]

  def post(self, request):
    return Response({
      'message': 'Cierre de sesión exitoso'
    })


class TokenRefreshView(APIView):

  def post(self, request):
    token = request.data.get('token')

    if not token:
      return Response({
        'error': 'missing_token',
        'error_description': 'Se requiere token'
      }, status=status.HTTP_400_BAD_REQUEST)

    new_token = JWTTokenManager.refresh_token(token)

    if new_token:
      return Response({
        'access_token': new_token,
        'token_type': 'Bearer',
        'expires_in': 86400
      })
    else:
      return Response({
        'error': 'invalid_token',
        'error_description': 'Token inválido o expirado'
      }, status=status.HTTP_401_UNAUTHORIZED)


def get_user_from_token(token: str) -> Optional[User]:
  payload = JWTTokenManager.validate_token(token)
  if payload and 'user_id' in payload:
    try:
      return User.objects.get(id=payload['user_id'])
    except User.DoesNotExist:
      pass
  return None


def require_authentication(view_func):
  def wrapper(request, *args, **kwargs):
    auth_header = request.META.get('HTTP_AUTHORIZATION')
    if not auth_header or not auth_header.startswith('Bearer '):
      return JsonResponse({
        'error': 'unauthorized',
        'error_description': 'Autenticación requerida'
      }, status=401)

    token = auth_header.split(' ')[1]
    user = get_user_from_token(token)
    if not user:
      return JsonResponse({
        'error': 'invalid_token',
        'error_description': 'Token inválido o expirado'
      }, status=401)

    request.user = user
    return view_func(request, *args, **kwargs)

  return wrapper
