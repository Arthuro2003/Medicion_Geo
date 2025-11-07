import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any

import redis
from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer
from django.conf import settings
from django.contrib.auth.models import User
from django.db import models
from django.utils import timezone


class UserSession(models.Model):
  SESSION_TYPES = [
    ('web', 'Web Browser'),
    ('api', 'API Client'),
    ('mobile', 'Mobile App'),
    ('desktop', 'Desktop App'),
  ]

  ACTIVITY_TYPES = [
    ('viewing', 'Viewing Project'),
    ('editing', 'Editing Image'),
    ('measuring', 'Making Measurements'),
    ('calibrating', 'Calibrating Image'),
    ('generating_report', 'Generating Report'),
    ('idle', 'Idle'),
    ('offline', 'Offline'),
  ]

  user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='user_sessions')
  session_key = models.CharField(max_length=40, unique=True)
  session_type = models.CharField(max_length=20, choices=SESSION_TYPES, default='web')

  ip_address = models.GenericIPAddressField()
  user_agent = models.TextField(blank=True)
  device_info = models.JSONField(default=dict)

  created_at = models.DateTimeField(auto_now_add=True)
  last_activity = models.DateTimeField(auto_now=True)
  is_active = models.BooleanField(default=True)

  current_activity = models.CharField(max_length=20, choices=ACTIVITY_TYPES, default='idle')
  current_project_id = models.UUIDField(null=True, blank=True)
  current_image_id = models.UUIDField(null=True, blank=True)
  activity_data = models.JSONField(default=dict)

  websocket_channel = models.CharField(max_length=255, blank=True)

  class Meta:
    verbose_name = "User Session"
    verbose_name_plural = "User Sessions"
    ordering = ['-last_activity']
    indexes = [
      models.Index(fields=['user', 'is_active']),
      models.Index(fields=['last_activity']),
      models.Index(fields=['current_project_id']),
    ]

  def __str__(self):
    return f"{self.user.username} - {self.session_type} ({self.ip_address})"

  def is_expired(self):
    return timezone.now() - self.last_activity > timedelta(minutes=30)

  def update_activity(self, activity_type: str, project_id: str = None,
                      image_id: str = None, data: Dict = None):
    self.current_activity = activity_type
    self.current_project_id = project_id
    self.current_image_id = image_id
    self.activity_data = data or {}
    self.last_activity = timezone.now()
    self.save(update_fields=['current_activity', 'current_project_id',
                             'current_image_id', 'activity_data', 'last_activity'])

  def get_activity_display_data(self):
    base_info = {
      'activity': self.get_current_activity_display(),
      'last_seen': self.last_activity.isoformat(),
      'duration': (timezone.now() - self.last_activity).total_seconds(),
    }

    if self.current_project_id:
      try:
        from core.models import Project
        project = Project.objects.get(id=self.current_project_id)
        base_info['project'] = project.name
      except Project.DoesNotExist:
        pass

    if self.current_image_id:
      try:
        from core.models import ProjectImage
        image = ProjectImage.objects.get(id=self.current_image_id)
        base_info['image'] = image.name
      except ProjectImage.DoesNotExist:
        pass

    return base_info


class ConcurrentUserManager:

  def __init__(self):
    self.logger = logging.getLogger('core.multi_user')
    self.redis_client = self._get_redis_client()

  def _get_redis_client(self):
    try:
      if hasattr(settings, 'REDIS_URL'):
        return redis.from_url(settings.REDIS_URL)
      else:
        return redis.Redis(
          host=getattr(settings, 'REDIS_HOST', 'localhost'),
          port=getattr(settings, 'REDIS_PORT', 6379),
          db=getattr(settings, 'REDIS_DB', 0),
          decode_responses=True
        )
    except:
      return None

  def create_user_session(self, user: User, request, session_type: str = 'web') -> UserSession:
    user_agent = request.META.get('HTTP_USER_AGENT', '')
    ip_address = self._get_client_ip(request)

    device_info = {
      'browser': self._parse_user_agent(user_agent),
      'platform': self._detect_platform(user_agent),
      'is_mobile': self._is_mobile(user_agent)
    }

    user_session = UserSession.objects.create(
      user=user,
      session_key=request.session.session_key or self._generate_session_key(),
      session_type=session_type,
      ip_address=ip_address,
      user_agent=user_agent,
      device_info=device_info
    )

    self._register_session_in_redis(user_session)

    self.logger.info(f"Created session for user {user.username} from {ip_address}")
    return user_session

  def _get_client_ip(self, request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
      ip = x_forwarded_for.split(',')[0]
    else:
      ip = request.META.get('REMOTE_ADDR')
    return ip

  def _generate_session_key(self):
    return str(uuid.uuid4()).replace('-', '')

  def _parse_user_agent(self, user_agent: str) -> str:
    ua_lower = user_agent.lower()
    if 'chrome' in ua_lower:
      return 'Chrome'
    elif 'firefox' in ua_lower:
      return 'Firefox'
    elif 'safari' in ua_lower:
      return 'Safari'
    elif 'edge' in ua_lower:
      return 'Edge'
    else:
      return 'Unknown'

  def _detect_platform(self, user_agent: str) -> str:
    ua_lower = user_agent.lower()
    if 'windows' in ua_lower:
      return 'Windows'
    elif 'mac' in ua_lower:
      return 'macOS'
    elif 'linux' in ua_lower:
      return 'Linux'
    elif 'android' in ua_lower:
      return 'Android'
    elif 'ios' in ua_lower or 'iphone' in ua_lower or 'ipad' in ua_lower:
      return 'iOS'
    else:
      return 'Unknown'

  def _is_mobile(self, user_agent: str) -> bool:
    mobile_keywords = ['mobile', 'android', 'iphone', 'ipad', 'tablet']
    return any(keyword in user_agent.lower() for keyword in mobile_keywords)

  def _register_session_in_redis(self, user_session: UserSession):
    if not self.redis_client:
      return

    session_data = {
      'user_id': user_session.user.id,
      'username': user_session.user.username,
      'session_type': user_session.session_type,
      'ip_address': user_session.ip_address,
      'created_at': user_session.created_at.isoformat(),
      'current_activity': user_session.current_activity,
    }

    session_key = f"session:{user_session.session_key}"
    self.redis_client.hset(session_key, mapping=session_data)
    self.redis_client.expire(session_key, 3600)

    self.redis_client.sadd("active_users", user_session.user.id)

    user_sessions_key = f"user_sessions:{user_session.user.id}"
    self.redis_client.sadd(user_sessions_key, user_session.session_key)
    self.redis_client.expire(user_sessions_key, 3600)

  def update_user_activity(self, session_key: str, activity_type: str,
                           project_id: str = None, image_id: str = None,
                           data: Dict = None) -> bool:
    try:
      user_session = UserSession.objects.get(session_key=session_key, is_active=True)
      user_session.update_activity(activity_type, project_id, image_id, data)

      if self.redis_client:
        session_redis_key = f"session:{session_key}"
        updates = {
          'current_activity': activity_type,
          'last_activity': timezone.now().isoformat()
        }
        if project_id:
          updates['current_project_id'] = project_id
        if image_id:
          updates['current_image_id'] = image_id

        self.redis_client.hset(session_redis_key, mapping=updates)
        self.redis_client.expire(session_redis_key, 3600)

      return True

    except UserSession.DoesNotExist:
      return False

  def get_active_users(self) -> List[Dict[str, Any]]:
    if self.redis_client:
      return self._get_active_users_from_redis()

    active_sessions = UserSession.objects.filter(
      is_active=True,
      last_activity__gte=timezone.now() - timedelta(minutes=30)
    ).select_related('user')

    users_data = []
    for session in active_sessions:
      users_data.append({
        'user_id': session.user.id,
        'username': session.user.username,
        'full_name': session.user.get_full_name(),
        'session_type': session.session_type,
        'ip_address': session.ip_address,
        'current_activity': session.get_current_activity_display(),
        'last_activity': session.last_activity.isoformat(),
        'device_info': session.device_info
      })

    return users_data

  def _get_active_users_from_redis(self) -> List[Dict[str, Any]]:
    try:
      active_user_ids = self.redis_client.smembers("active_users")
      users_data = []

      for user_id in active_user_ids:
        user_sessions_key = f"user_sessions:{user_id}"
        session_keys = self.redis_client.smembers(user_sessions_key)

        for session_key in session_keys:
          session_data = self.redis_client.hgetall(f"session:{session_key}")
          if session_data:
            users_data.append({
              'user_id': int(session_data['user_id']),
              'username': session_data['username'],
              'session_type': session_data['session_type'],
              'ip_address': session_data['ip_address'],
              'current_activity': session_data.get('current_activity', 'idle'),
              'last_activity': session_data.get('last_activity', ''),
              'session_key': session_key
            })

      return users_data

    except Exception as e:
      self.logger.error(f"Error getting active users from Redis: {str(e)}")
      return []

  def get_project_collaborators(self, project_id: str) -> List[Dict[str, Any]]:
    if self.redis_client:
      active_users = self._get_active_users_from_redis()
      collaborators = [
        user for user in active_users
        if user.get('current_project_id') == project_id
      ]
    else:
      sessions = UserSession.objects.filter(
        current_project_id=project_id,
        is_active=True,
        last_activity__gte=timezone.now() - timedelta(minutes=30)
      ).select_related('user')

      collaborators = []
      for session in sessions:
        collaborators.append(session.get_activity_display_data())

    return collaborators

  def check_resource_lock(self, resource_type: str, resource_id: str,
                          user_id: int, session_key: str) -> Dict[str, Any]:
    lock_key = f"lock:{resource_type}:{resource_id}"

    if self.redis_client:
      existing_lock = self.redis_client.hgetall(lock_key)

      if existing_lock:
        if (existing_lock.get('user_id') == str(user_id) and
          existing_lock.get('session_key') == session_key):
          self.redis_client.expire(lock_key, 300)
          return {
            'locked': True,
            'owned_by_user': True,
            'lock_data': existing_lock
          }
        else:
          return {
            'locked': True,
            'owned_by_user': False,
            'lock_data': existing_lock,
            'locked_by': existing_lock.get('username'),
            'locked_since': existing_lock.get('locked_at')
          }
      else:
        return {
          'locked': False,
          'can_acquire': True
        }

    return self._check_resource_lock_db(resource_type, resource_id, user_id)

  def acquire_resource_lock(self, resource_type: str, resource_id: str,
                            user_id: int, session_key: str,
                            lock_duration: int = 300) -> bool:

    if not self.redis_client:
      return True

    lock_key = f"lock:{resource_type}:{resource_id}"

    lock_check = self.check_resource_lock(resource_type, resource_id, user_id, session_key)

    if lock_check['locked'] and not lock_check.get('owned_by_user', False):
      return False

    user = User.objects.get(id=user_id)
    lock_data = {
      'user_id': user_id,
      'username': user.username,
      'session_key': session_key,
      'resource_type': resource_type,
      'resource_id': resource_id,
      'locked_at': timezone.now().isoformat()
    }

    self.redis_client.hset(lock_key, mapping=lock_data)
    self.redis_client.expire(lock_key, lock_duration)

    self.logger.info(f"Lock acquired on {resource_type}:{resource_id} by user {user.username}")
    return True

  def release_resource_lock(self, resource_type: str, resource_id: str,
                            user_id: int, session_key: str) -> bool:
    if not self.redis_client:
      return True

    lock_key = f"lock:{resource_type}:{resource_id}"

    existing_lock = self.redis_client.hgetall(lock_key)
    if (existing_lock and
      existing_lock.get('user_id') == str(user_id) and
      existing_lock.get('session_key') == session_key):
      self.redis_client.delete(lock_key)
      self.logger.info(f"Lock released on {resource_type}:{resource_id} by user {user_id}")
      return True

    return False

  def cleanup_expired_sessions(self):
    expired_sessions = UserSession.objects.filter(
      last_activity__lt=timezone.now() - timedelta(minutes=30),
      is_active=True
    )

    for session in expired_sessions:
      session.is_active = False
      session.current_activity = 'offline'
      session.save()

      if self.redis_client:
        self.redis_client.srem("active_users", session.user.id)
        self.redis_client.delete(f"session:{session.session_key}")
        self.redis_client.srem(f"user_sessions:{session.user.id}", session.session_key)

    if self.redis_client:
      self._cleanup_expired_locks()

    self.logger.info(f"Cleaned up {expired_sessions.count()} expired sessions")

  def _cleanup_expired_locks(self):
    try:
      lock_keys = self.redis_client.keys("lock:*")

      for lock_key in lock_keys:
        if not self.redis_client.exists(lock_key):
          continue

        lock_data = self.redis_client.hgetall(lock_key)
        if lock_data:
          locked_at = datetime.fromisoformat(lock_data.get('locked_at', ''))
          if timezone.now() - locked_at > timedelta(minutes=10):
            self.redis_client.delete(lock_key)
            self.logger.info(f"Force cleaned expired lock: {lock_key}")

    except Exception as e:
      self.logger.error(f"Error cleaning expired locks: {str(e)}")

  def get_user_session_count(self) -> int:
    if self.redis_client:
      return len(self.redis_client.smembers("active_users"))

    return UserSession.objects.filter(
      is_active=True,
      last_activity__gte=timezone.now() - timedelta(minutes=30)
    ).count()

  def get_concurrent_limit(self) -> int:
    return getattr(settings, 'MAX_CONCURRENT_USERS', 100)

  def can_create_new_session(self) -> bool:
    current_sessions = self.get_user_session_count()
    return current_sessions < self.get_concurrent_limit()

  def terminate_user_session(self, session_key: str, reason: str = "Admin termination"):
    try:
      user_session = UserSession.objects.get(session_key=session_key)
      user_session.is_active = False
      user_session.current_activity = 'offline'
      user_session.activity_data = {'termination_reason': reason}
      user_session.save()

      if self.redis_client:
        self.redis_client.srem("active_users", user_session.user.id)
        self.redis_client.delete(f"session:{session_key}")
        self.redis_client.srem(f"user_sessions:{user_session.user.id}", session_key)

      self._release_all_session_locks(user_session.user.id, session_key)

      self.logger.info(f"Terminated session {session_key}: {reason}")

    except UserSession.DoesNotExist:
      pass

  def _release_all_session_locks(self, user_id: int, session_key: str):
    if not self.redis_client:
      return

    try:
      lock_keys = self.redis_client.keys("lock:*")

      for lock_key in lock_keys:
        lock_data = self.redis_client.hgetall(lock_key)
        if (lock_data and
          lock_data.get('user_id') == str(user_id) and
          lock_data.get('session_key') == session_key):
          self.redis_client.delete(lock_key)
          self.logger.info(f"Released lock {lock_key} for terminated session")

    except Exception as e:
      self.logger.error(f"Error releasing session locks: {str(e)}")


class UserActivityConsumer(AsyncWebsocketConsumer):
  async def connect(self):
    self.user = self.scope["user"]
    if not self.user.is_authenticated:
      await self.close()
      return

    self.user_group_name = f"user_{self.user.id}"
    await self.channel_layer.group_add(self.user_group_name, self.channel_name)
    await self.channel_layer.group_add("global_activity", self.channel_name)
    await self.update_user_websocket_channel()
    await self.accept()
    await self.send_user_status()

  async def disconnect(self, close_code):
    await self.channel_layer.group_discard(self.user_group_name, self.channel_name)
    await self.channel_layer.group_discard("global_activity", self.channel_name)
    await self.clear_user_websocket_channel()

  async def receive(self, text_data=None, bytes_data=None):
    if text_data is None:
      return

    try:
      data = json.loads(text_data)
      message_type = data.get('type')

      if message_type == 'activity_update':
        await self.handle_activity_update(data)
      elif message_type == 'join_project':
        await self.handle_join_project(data)
      elif message_type == 'leave_project':
        await self.handle_leave_project(data)
      elif message_type == 'request_lock':
        await self.handle_lock_request(data)
      elif message_type == 'release_lock':
        await self.handle_lock_release(data)

    except json.JSONDecodeError:
      await self.send(text_data=json.dumps({
        'error': 'Formato JSON inválido'
      }))

  async def handle_activity_update(self, data: Dict) -> None:
    manager = ConcurrentUserManager()
    success = await database_sync_to_async(manager.update_user_activity)(
      session_key=self.scope["session"].session_key,
      activity_type=data.get('activity'),
      project_id=data.get('project_id'),
      image_id=data.get('image_id'),
      data=data.get('data', {})
    )

    if success and data.get('project_id'):
      await self.channel_layer.group_send(
        f"project_{data['project_id']}",
        {
          'type': 'user_activity_update',
          'user_id': self.user.id,
          'username': self.user.username,
          'activity': data.get('activity'),
          'data': data.get('data', {})
        }
      )

  async def handle_join_project(self, data: Dict) -> None:
    project_id = data.get('project_id')
    if project_id:
      await self.channel_layer.group_add(f"project_{project_id}", self.channel_name)
      await self.channel_layer.group_send(
        f"project_{project_id}",
        {
          'type': 'user_joined_project',
          'user_id': self.user.id,
          'username': self.user.username
        }
      )

  async def handle_leave_project(self, data: Dict) -> None:
    project_id = data.get('project_id')
    if project_id:
      await self.channel_layer.group_discard(f"project_{project_id}", self.channel_name)
      await self.channel_layer.group_send(
        f"project_{project_id}",
        {
          'type': 'user_left_project',
          'user_id': self.user.id,
          'username': self.user.username
        }
      )

  async def handle_lock_request(self, data: Dict) -> None:
    manager = ConcurrentUserManager()
    success = await database_sync_to_async(manager.acquire_resource_lock)(
      resource_type=data.get('resource_type'),
      resource_id=data.get('resource_id'),
      user_id=self.user.id,
      session_key=self.scope["session"].session_key
    )

    await self.send(text_data=json.dumps({
      'type': 'lock_response',
      'resource_type': data.get('resource_type'),
      'resource_id': data.get('resource_id'),
      'success': success,
      'locked_by_user': success
    }))

    if success and data.get('project_id'):
      await self.channel_layer.group_send(
        f"project_{data['project_id']}",
        {
          'type': 'resource_locked',
          'resource_type': data.get('resource_type'),
          'resource_id': data.get('resource_id'),
          'locked_by': self.user.username,
          'user_id': self.user.id
        }
      )

  async def handle_lock_release(self, data: Dict) -> None:
    manager = ConcurrentUserManager()
    success = await database_sync_to_async(manager.release_resource_lock)(
      resource_type=data.get('resource_type'),
      resource_id=data.get('resource_id'),
      user_id=self.user.id,
      session_key=self.scope["session"].session_key
    )

    if success and data.get('project_id'):
      await self.channel_layer.group_send(
        f"project_{data['project_id']}",
        {
          'type': 'resource_unlocked',
          'resource_type': data.get('resource_type'),
          'resource_id': data.get('resource_id'),
          'released_by': self.user.username
        }
      )

  async def user_activity_update(self, event: Dict) -> None:
    await self.send(text_data=json.dumps({
      'type': 'activity_update',
      'user_id': event['user_id'],
      'username': event['username'],
      'activity': event['activity'],
      'data': event.get('data', {})
    }))

  async def user_joined_project(self, event: Dict) -> None:
    await self.send(text_data=json.dumps({
      'type': 'user_joined',
      'user_id': event['user_id'],
      'username': event['username']
    }))

  async def user_left_project(self, event: Dict) -> None:
    await self.send(text_data=json.dumps({
      'type': 'user_left',
      'user_id': event['user_id'],
      'username': event['username']
    }))

  async def resource_locked(self, event: Dict) -> None:
    if event['user_id'] != self.user.id:
      await self.send(text_data=json.dumps({
        'type': 'resource_locked',
        'resource_type': event['resource_type'],
        'resource_id': event['resource_id'],
        'locked_by': event['locked_by']
      }))

  async def resource_unlocked(self, event: Dict) -> None:
    await self.send(text_data=json.dumps({
      'type': 'resource_unlocked',
      'resource_type': event['resource_type'],
      'resource_id': event['resource_id'],
      'released_by': event['released_by']
    }))

  @database_sync_to_async
  def update_user_websocket_channel(self) -> None:
    try:
      user_session = UserSession.objects.get(
        session_key=self.scope["session"].session_key,
        user=self.user,
        is_active=True
      )
      user_session.websocket_channel = self.channel_name
      user_session.save(update_fields=['websocket_channel'])
    except UserSession.DoesNotExist:
      pass

  @database_sync_to_async
  def clear_user_websocket_channel(self) -> None:
    try:
      user_session = UserSession.objects.get(
        websocket_channel=self.channel_name,
        user=self.user
      )
      user_session.websocket_channel = ''
      user_session.save(update_fields=['websocket_channel'])
    except UserSession.DoesNotExist:
      pass

  async def send_user_status(self) -> None:
    await self.send(text_data=json.dumps({
      'type': 'user_status',
      'user_id': self.user.id,
      'username': self.user.username,
      'status': 'en línea'
    }))
