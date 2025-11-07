import json
import logging
import shutil
import threading
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import schedule
from django.conf import settings
from django.contrib.auth.models import User
from django.core.mail import send_mail
from django.core.management import call_command

from .storage_backends import CloudStorageBackend, LocalStorageBackend

DEFAULT_BACKUP_SETTINGS = {
  'enabled': True,
  'schedule_daily': True,
  'schedule_weekly': True,
  'schedule_monthly': True,
  'retention_days': 30,
  'retention_weeks': 12,
  'retention_months': 12,
  'compress': True,
  'encrypt': False,
  'email_notifications': True,
  'max_backup_size_mb': 1000,
  'backup_types': ['database', 'media', 'reports', 'logs']
}


class BackupManager:

  def __init__(self):
    self.backup_dir = Path(settings.MEDIA_ROOT) / 'backups'
    self.backup_dir.mkdir(exist_ok=True)
    self.logger = logging.getLogger('core.backup')
    self.storage_backends = self._initialize_storage_backends()
    self.config = getattr(settings, 'BACKUP_SETTINGS', DEFAULT_BACKUP_SETTINGS)

    self.backup_thread = None
    self.scheduler_running = False

  def _initialize_storage_backends(self) -> Dict:
    return {
      'local': LocalStorageBackend(str(self.backup_dir)),
      'cloud': CloudStorageBackend() if hasattr(settings, 'CLOUD_STORAGE_ENABLED') else None
    }

  def start_scheduler(self) -> None:
    if not self.config['enabled'] or self.scheduler_running:
      return

    self.logger.info("Iniciando programador de respaldos")

    if self.config['schedule_daily']:
      schedule.every().day.at("02:00").do(self._run_daily_backup)

    if self.config['schedule_weekly']:
      schedule.every().sunday.at("03:00").do(self._run_weekly_backup)

    if self.config['schedule_monthly']:
      schedule.every().month.do(self._run_monthly_backup)

    self.scheduler_running = True
    self.backup_thread = threading.Thread(target=self._scheduler_worker, daemon=True)
    self.backup_thread.start()

  def stop_scheduler(self):
    self.scheduler_running = False
    schedule.clear()
    self.logger.info("El programador de copias de seguridad se detuvo")

  def _scheduler_worker(self) -> None:
    while self.scheduler_running:
      try:
        schedule.run_pending()
        time.sleep(60)
      except Exception as e:
        self.logger.error(f"Error en el programador: {str(e)}")
        time.sleep(300)

  def _run_daily_backup(self) -> None:
    try:
      self.logger.info("Iniciando respaldo diario")
      result = self.create_backup(
        backup_type='daily',
        include_types=['database', 'reports']
      )
      self._send_notification('daily', result['success'], result)
    except Exception as e:
      self.logger.error(f"Falló el respaldo diario: {str(e)}")
      self._send_notification('daily', False, {'error': str(e)})

  def _run_weekly_backup(self):
    try:
      self.logger.info("Comenzando el respaldo semanal.")
      result = self.create_backup(
        backup_type='weekly',
        include_types=['database', 'media', 'reports']
      )
      if result['success']:
        self._send_notification('weekly', True, result)
        self._cleanup_old_backups('daily', days=7)
      else:
        self._send_notification('weekly', False, result)
    except Exception as e:
      self.logger.error(f"La copia de seguridad semanal falló {str(e)}")
      self._send_notification('weekly', False, {'error': str(e)})

  def _run_monthly_backup(self):
    try:
      self.logger.info("Comenzando el respaldo mensual.\"")
      result = self.create_backup(
        backup_type='monthly',
        include_types=['database', 'media', 'reports', 'logs']
      )
      if result['success']:
        self._send_notification('monthly', True, result)
        self._cleanup_old_backups('weekly', days=30)
      else:
        self._send_notification('monthly', False, result)
    except Exception as e:
      self.logger.error(f"Monthly backup failed: {str(e)}")
      self._send_notification('monthly', False, {'error': str(e)})

  def create_backup(self,
                    backup_type: str = 'manual',
                    include_types: List[str] = None,
                    compress: bool = None,
                    storage_backend: str = 'local') -> Dict[str, Any]:

    if include_types is None:
      include_types = self.config['backup_types']

    if compress is None:
      compress = self.config['compress']

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_name = f"backup_{backup_type}_{timestamp}"
    backup_path = self.backup_dir / backup_name
    backup_path.mkdir(exist_ok=True)

    try:
      backup_info = {
        'name': backup_name,
        'type': backup_type,
        'timestamp': timestamp,
        'datetime': datetime.now().isoformat(),
        'included_types': include_types,
        'compressed': compress,
        'files': [],
        'size_bytes': 0,
        'success': True,
        'errors': []
      }

      if 'database' in include_types:
        db_result = self._backup_database(backup_path)
        backup_info['files'].extend(db_result['files'])
        backup_info['size_bytes'] += db_result['size']
        if not db_result['success']:
          backup_info['errors'].extend(db_result['errors'])

      if 'media' in include_types:
        media_result = self._backup_media_files(backup_path)
        backup_info['files'].extend(media_result['files'])
        backup_info['size_bytes'] += media_result['size']
        if not media_result['success']:
          backup_info['errors'].extend(media_result['errors'])

      if 'reports' in include_types:
        reports_result = self._backup_reports(backup_path)
        backup_info['files'].extend(reports_result['files'])
        backup_info['size_bytes'] += reports_result['size']
        if not reports_result['success']:
          backup_info['errors'].extend(reports_result['errors'])

      if 'logs' in include_types:
        logs_result = self._backup_logs(backup_path)
        backup_info['files'].extend(logs_result['files'])
        backup_info['size_bytes'] += logs_result['size']
        if not logs_result['success']:
          backup_info['errors'].extend(logs_result['errors'])

      manifest_path = backup_path / 'manifest.json'
      with open(manifest_path, 'w') as f:
        json.dump(backup_info, f, indent=2)

      backup_info['files'].append(str(manifest_path))

      if compress:
        compressed_result = self._compress_backup(backup_path)
        if compressed_result['success']:
          backup_info['compressed_file'] = compressed_result['file']
          backup_info['compressed_size'] = compressed_result['size']
          shutil.rmtree(backup_path)
        else:
          backup_info['errors'].extend(compressed_result['errors'])

      if storage_backend == 'cloud' and self.storage_backends['cloud']:
        cloud_result = self._upload_to_cloud(backup_info)
        if cloud_result['success']:
          backup_info['cloud_location'] = cloud_result['location']
        else:
          backup_info['errors'].extend(cloud_result['errors'])

      if backup_info['errors']:
        backup_info['success'] = False
        self.logger.warning(f"Backup {backup_name} completed with errors: {backup_info['errors']}")
      else:
        self.logger.info(f"Backup {backup_name} completed successfully")

      size_mb = backup_info['size_bytes'] / (1024 * 1024)
      if size_mb > self.config['max_backup_size_mb']:
        backup_info['warnings'] = backup_info.get('warnings', [])
        backup_info['warnings'].append(
          f"Backup size ({size_mb:.1f}MB) exceeds limit ({self.config['max_backup_size_mb']}MB)")

      return backup_info

    except Exception as e:
      self.logger.error(f"Backup creation failed: {str(e)}")
      return {
        'success': False,
        'error': str(e),
        'name': backup_name,
        'type': backup_type,
        'timestamp': timestamp
      }

  def _backup_database(self, backup_path: Path) -> Dict[str, Any]:
    try:
      db_file = backup_path / 'database.json'

      with open(db_file, 'w') as f:
        call_command('dumpdata',
                     exclude=['contenttypes', 'sessions', 'admin.logentry'],
                     indent=2,
                     stdout=f)

      size = db_file.stat().st_size

      return {
        'success': True,
        'files': [str(db_file)],
        'size': size,
        'errors': []
      }

    except Exception as e:
      return {
        'success': False,
        'files': [],
        'size': 0,
        'errors': [f"Database backup failed: {str(e)}"]
      }

  def _backup_media_files(self, backup_path: Path) -> Dict[str, Any]:
    try:
      media_dir = backup_path / 'media'
      media_dir.mkdir(exist_ok=True)

      source_media = Path(settings.MEDIA_ROOT)
      total_size = 0
      files = []

      if source_media.exists():
        for item in source_media.rglob('*'):
          if item.is_file() and not item.name.startswith('.'):
            relative_path = item.relative_to(source_media)
            dest_path = media_dir / relative_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(item, dest_path)
            total_size += item.stat().st_size
            files.append(str(dest_path))

      return {
        'success': True,
        'files': files,
        'size': total_size,
        'errors': []
      }

    except Exception as e:
      return {
        'success': False,
        'files': [],
        'size': 0,
        'errors': [f"Media backup failed: {str(e)}"]
      }

  def _backup_reports(self, backup_path: Path) -> Dict[str, Any]:
    try:
      reports_dir = backup_path / 'reports'
      reports_dir.mkdir(exist_ok=True)

      source_reports = Path(settings.MEDIA_ROOT) / 'reports'
      total_size = 0
      files = []

      if source_reports.exists():
        for report_file in source_reports.rglob('*'):
          if report_file.is_file():
            relative_path = report_file.relative_to(source_reports)
            dest_path = reports_dir / relative_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(report_file, dest_path)
            total_size += report_file.stat().st_size
            files.append(str(dest_path))

      return {
        'success': True,
        'files': files,
        'size': total_size,
        'errors': []
      }

    except Exception as e:
      return {
        'success': False,
        'files': [],
        'size': 0,
        'errors': [f"Reports backup failed: {str(e)}"]
      }

  def _backup_logs(self, backup_path: Path) -> Dict[str, Any]:
    try:
      logs_dir = backup_path / 'logs'
      logs_dir.mkdir(exist_ok=True)

      logs_source = Path(settings.BASE_DIR) / 'logs'
      total_size = 0
      files = []

      if logs_source.exists():
        for log_file in logs_source.glob('*.log'):
          if log_file.is_file():
            dest_path = logs_dir / log_file.name
            shutil.copy2(log_file, dest_path)
            total_size += log_file.stat().st_size
            files.append(str(dest_path))

      return {
        'success': True,
        'files': files,
        'size': total_size,
        'errors': []
      }

    except Exception as e:
      return {
        'success': False,
        'files': [],
        'size': 0,
        'errors': [f"Logs backup failed: {str(e)}"]
      }

  def _compress_backup(self, backup_path: Path) -> Dict[str, Any]:
    try:
      zip_path = backup_path.with_suffix('.zip')

      with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in backup_path.rglob('*'):
          if file_path.is_file():
            arcname = file_path.relative_to(backup_path.parent)
            zipf.write(file_path, arcname)

      size = zip_path.stat().st_size

      return {
        'success': True,
        'file': str(zip_path),
        'size': size,
        'errors': []
      }

    except Exception as e:
      return {
        'success': False,
        'file': None,
        'size': 0,
        'errors': [f"Compression failed: {str(e)}"]
      }

  def _upload_to_cloud(self, backup_info: Dict[str, Any]) -> Dict[str, Any]:
    try:
      if not self.storage_backends['cloud']:
        return {
          'success': False,
          'errors': ['Cloud storage not configured']
        }

      file_path = backup_info.get('compressed_file') or backup_info['files'][0]
      result = self.storage_backends['cloud'].upload(file_path, backup_info['name'])

      return {
        'success': result['success'],
        'location': result.get('location'),
        'errors': result.get('errors', [])
      }

    except Exception as e:
      return {
        'success': False,
        'errors': [f"Cloud upload failed: {str(e)}"]
      }

  def _cleanup_old_backups(self, backup_type: str, days: int):
    try:
      cutoff_date = datetime.now() - timedelta(days=days)

      pattern = f"backup_{backup_type}_*"
      for backup_item in self.backup_dir.glob(pattern):
        try:
          timestamp_str = backup_item.stem.split('_')[-1]
          backup_date = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')

          if backup_date < cutoff_date:
            if backup_item.is_dir():
              shutil.rmtree(backup_item)
            else:
              backup_item.unlink()
            self.logger.info(f"Removed old backup: {backup_item.name}")

        except (ValueError, IndexError):
          continue

    except Exception as e:
      self.logger.error(f"Cleanup failed for {backup_type} backups: {str(e)}")

  def _send_notification(self, backup_type: str, success: bool, result: Dict[str, Any]):
    if not self.config['email_notifications']:
      return

    try:
      admin_users = User.objects.filter(is_staff=True, is_active=True)
      admin_emails = [user.email for user in admin_users if user.email]

      if not admin_emails:
        return

      status = "EXITOSO" if success else "FALLIDO"
      subject = f"Backup {backup_type.title()} - {status}"

      if success:
        size_mb = result.get('size_bytes', 0) / (1024 * 1024)
        message = f"""
                El backup {backup_type} se completó exitosamente.

                Detalles:
                - Nombre: {result.get('name', 'N/A')}
                - Tamaño: {size_mb:.2f} MB
                - Archivos incluidos: {len(result.get('files', []))}
                - Fecha: {result.get('datetime', 'N/A')}

                El backup está disponible en el sistema de almacenamiento configurado.
                """
      else:
        message = f"""
                El backup {backup_type} FALLÓ.

                Error:
                {result.get('error', 'Error desconocido')}

                Errores adicionales:
                {chr(10).join(result.get('errors', []))}

                Por favor, revise la configuración del sistema de backup.
                """

      send_mail(
        subject=subject,
        message=message,
        from_email=settings.DEFAULT_FROM_EMAIL,
        recipient_list=admin_emails,
        fail_silently=True
      )

    except Exception as e:
      self.logger.error(f"No se pudo enviar la notificación del respaldo {str(e)}")

  def restore_backup(self, backup_name: str) -> Dict[str, Any]:

    try:
      backup_path = self.backup_dir / backup_name

      if not backup_path.exists():
        zip_path = backup_path.with_suffix('.zip')
        if zip_path.exists():
          with zipfile.ZipFile(zip_path, 'r') as zipf:
            zipf.extractall(backup_path.parent)
          backup_path = self.backup_dir / backup_name
        else:
          raise FileNotFoundError(f"No se encontró el respaldo {backup_name}.")

      manifest_path = backup_path / 'manifest.json'
      if not manifest_path.exists():
        raise FileNotFoundError("No se encontró el archivo de manifiesto del respaldo.")

      with open(manifest_path, 'r') as f:
        backup_info = json.load(f)

      restore_results = {
        'success': True,
        'backup_name': backup_name,
        'restored_types': [],
        'errors': []
      }

      if 'database' in backup_info['included_types']:
        db_result = self._restore_database(backup_path)
        if db_result['success']:
          restore_results['restored_types'].append('database')
        else:
          restore_results['errors'].extend(db_result['errors'])

      if 'media' in backup_info['included_types']:
        media_result = self._restore_media_files(backup_path)
        if media_result['success']:
          restore_results['restored_types'].append('media')
        else:
          restore_results['errors'].extend(media_result['errors'])

      if restore_results['errors']:
        restore_results['success'] = False

      return restore_results

    except Exception as e:
      return {
        'success': False,
        'error': str(e),
        'backup_name': backup_name
      }

  def _restore_database(self, backup_path: Path) -> Dict[str, Any]:
    try:
      db_file = backup_path / 'database.json'
      if not db_file.exists():
        return {
          'success': False,
          'errors': ['Database backup file not found']
        }

      call_command('loaddata', str(db_file))

      return {
        'success': True,
        'errors': []
      }

    except Exception as e:
      return {
        'success': False,
        'errors': [f"Database restore failed: {str(e)}"]
      }

  def _restore_media_files(self, backup_path: Path) -> Dict[str, Any]:
    try:
      media_backup_dir = backup_path / 'media'
      if not media_backup_dir.exists():
        return {
          'success': False,
          'errors': ['Media backup directory not found']
        }

      media_dir = Path(settings.MEDIA_ROOT)

      for item in media_backup_dir.rglob('*'):
        if item.is_file():
          relative_path = item.relative_to(media_backup_dir)
          dest_path = media_dir / relative_path
          dest_path.parent.mkdir(parents=True, exist_ok=True)
          shutil.copy2(item, dest_path)

      return {
        'success': True,
        'errors': []
      }

    except Exception as e:
      return {
        'success': False,
        'errors': [f"Media restore failed: {str(e)}"]
      }

  def list_available_backups(self) -> List[Dict[str, Any]]:
    backups = []

    for backup_item in self.backup_dir.iterdir():
      if backup_item.name.startswith('backup_'):
        try:
          backup_info = {
            'name': backup_item.name,
            'path': str(backup_item),
            'is_compressed': backup_item.suffix == '.zip'
          }

          try:
            timestamp_str = backup_item.stem.split('_')[-1]
            backup_date = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
            backup_info['created'] = backup_date.isoformat()
            backup_info['created_display'] = backup_date.strftime('%Y-%m-%d %H:%M:%S')
          except (ValueError, IndexError):
            backup_info['created'] = None
            backup_info['created_display'] = 'Unknown'

          if backup_item.is_file():
            backup_info['size_bytes'] = backup_item.stat().st_size
          elif backup_item.is_dir():
            backup_info['size_bytes'] = sum(f.stat().st_size for f in backup_item.rglob('*') if f.is_file())
          else:
            backup_info['size_bytes'] = 0

          backup_info['size_mb'] = backup_info['size_bytes'] / (1024 * 1024)

          if backup_item.is_dir():
            manifest_path = backup_item / 'manifest.json'
            if manifest_path.exists():
              try:
                with open(manifest_path, 'r') as f:
                  manifest = json.load(f)
                backup_info['type'] = manifest.get('type', 'unknown')
                backup_info['included_types'] = manifest.get('included_types', [])
              except:
                backup_info['type'] = 'unknown'
                backup_info['included_types'] = []
          else:
            backup_info['type'] = 'compressed'
            backup_info['included_types'] = ['unknown']

          backups.append(backup_info)

        except Exception as e:
          self.logger.warning(f"No se pudo procesar el respaldo. {backup_item.name}: {str(e)}")

    backups.sort(key=lambda x: x.get('created', ''), reverse=True)
    return backups

  def get_backup_status(self) -> Dict[str, Any]:
    return {
      'scheduler_running': self.scheduler_running,
      'config': self.config,
      'available_backups': len(self.list_available_backups()),
      'backup_dir': str(self.backup_dir),
      'backup_dir_size_mb': sum(f.stat().st_size for f in self.backup_dir.rglob('*') if f.is_file()) / (1024 * 1024),
      'next_scheduled_backup': self._get_next_scheduled_backup()
    }

  def _get_next_scheduled_backup(self) -> Optional[str]:
    try:
      jobs = schedule.jobs
      if jobs:
        next_job = min(jobs, key=lambda job: job.next_run)
        return next_job.next_run.strftime('%Y-%m-%d %H:%M:%S')
    except:
      pass
    return None
