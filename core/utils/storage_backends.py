import logging
from pathlib import Path
from typing import Dict, Any

import boto3
from django.conf import settings


class StorageBackend:
  def __init__(self):
    self.logger = logging.getLogger('core.storage')

  def upload(self, file_path: str, backup_name: str) -> Dict[str, Any]:
    raise NotImplementedError

  def download(self, storage_path: str, local_path: str) -> Dict[str, Any]:
    raise NotImplementedError

  def delete(self, storage_path: str) -> Dict[str, Any]:
    raise NotImplementedError


class LocalStorageBackend(StorageBackend):
  def __init__(self, base_path: str):
    super().__init__()
    self.base_path = Path(base_path)
    self.base_path.mkdir(parents=True, exist_ok=True)

  def upload(self, file_path: str, backup_name: str) -> Dict[str, Any]:
    try:
      source_path = Path(file_path)
      dest_path = self.base_path / backup_name
      return self._handle_local_copy(source_path, dest_path)
    except Exception as e:
      self.logger.error(f"Error al subir archivo localmente: {str(e)}")
      return {'success': False, 'error': str(e)}

  def download(self, storage_path: str, local_path: str) -> Dict[str, Any]:
    try:
      source_path = self.base_path / storage_path
      dest_path = Path(local_path)
      dest_path.parent.mkdir(parents=True, exist_ok=True)
      return self._handle_local_copy(source_path, dest_path, is_download=True)
    except Exception as e:
      self.logger.error(f"Error al descargar archivo localmente: {str(e)}")
      return {'success': False, 'error': str(e)}

  def delete(self, storage_path: str) -> Dict[str, Any]:
    try:
      target_path = self.base_path / storage_path
      return self._handle_local_delete(target_path)
    except Exception as e:
      self.logger.error(f"Error al eliminar archivo localmente: {str(e)}")
      return {'success': False, 'error': str(e)}

  def _handle_local_copy(self, source_path: Path, dest_path: Path, is_download: bool = False) -> Dict[str, Any]:
    import shutil
    if source_path.is_file():
      shutil.copy2(source_path, dest_path)
    elif source_path.is_dir():
      shutil.copytree(source_path, dest_path)
    else:
      error_key = 'local_path' if is_download else 'location'
      return {
        'success': False,
        'error': f"La ruta no existe: {source_path}"
      }
    return {
      'success': True,
      'location' if not is_download else 'local_path': str(dest_path),
      **({'size': self._get_size(dest_path)} if not is_download else {})
    }

  def _handle_local_delete(self, target_path: Path) -> Dict[str, Any]:
    import shutil
    if target_path.exists():
      if target_path.is_file():
        target_path.unlink()
      else:
        shutil.rmtree(target_path)
      return {'success': True}
    return {'success': False, 'error': f"La ruta no existe: {target_path}"}

  def _get_size(self, path: Path) -> int:
    if path.is_file():
      return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())


class CloudStorageBackend(StorageBackend):
  def __init__(self):
    super().__init__()
    self.bucket_name = getattr(settings, 'AWS_STORAGE_BUCKET_NAME', 'medicion-backups')
    self.region = getattr(settings, 'AWS_S3_REGION_NAME', 'us-east-1')
    self.s3_client = self._initialize_s3_client()

  def upload(self, file_path: str, backup_name: str) -> Dict[str, Any]:
    if not self.s3_client:
      return {'success': False, 'error': 'Cliente S3 no configurado'}
    try:
      source_path = Path(file_path)
      if source_path.is_file():
        return self._upload_file(source_path, backup_name)
      elif source_path.is_dir():
        return self._upload_directory(source_path, backup_name)
      return {'success': False, 'error': f"La ruta no existe: {file_path}"}
    except Exception as e:
      self.logger.error(f"Error al subir a S3: {str(e)}")
      return {'success': False, 'error': str(e)}

  def download(self, storage_path: str, local_path: str) -> Dict[str, Any]:
    if not self.s3_client:
      return {'success': False, 'error': 'Cliente S3 no configurado'}
    try:
      cleaned_path = self._clean_s3_path(storage_path)
      dest_path = Path(local_path)
      dest_path.parent.mkdir(parents=True, exist_ok=True)
      with open(dest_path, 'wb') as f:
        self.s3_client.download_fileobj(self.bucket_name, cleaned_path, f)
      return {'success': True, 'local_path': str(dest_path)}
    except Exception as e:
      self.logger.error(f"Error al descargar desde S3: {str(e)}")
      return {'success': False, 'error': str(e)}

  def delete(self, storage_path: str) -> Dict[str, Any]:
    if not self.s3_client:
      return {'success': False, 'error': 'Cliente S3 no configurado'}
    try:
      cleaned_path = self._clean_s3_path(storage_path)
      if cleaned_path.endswith('/'):
        self._delete_s3_directory(cleaned_path)
      else:
        self.s3_client.delete_object(Bucket=self.bucket_name, Key=cleaned_path)
      return {'success': True}
    except Exception as e:
      self.logger.error(f"Error al eliminar en S3: {str(e)}")
      return {'success': False, 'error': str(e)}

  def _initialize_s3_client(self):
    try:
      return boto3.client(
        's3',
        aws_access_key_id=getattr(settings, 'AWS_ACCESS_KEY_ID', None),
        aws_secret_access_key=getattr(settings, 'AWS_SECRET_ACCESS_KEY', None),
        region_name=self.region
      )
    except Exception as e:
      self.logger.warning(f"Error al inicializar cliente S3: {str(e)}")
      return None

  def _upload_file(self, source_path: Path, backup_name: str) -> Dict[str, Any]:
    s3_key = f"backups/{backup_name}/{source_path.name}"
    with open(source_path, 'rb') as f:
      self.s3_client.upload_fileobj(f, self.bucket_name, s3_key)
    return {
      'success': True,
      'location': f"s3://{self.bucket_name}/{s3_key}",
      'size': source_path.stat().st_size
    }

  def _upload_directory(self, source_path: Path, backup_name: str) -> Dict[str, Any]:
    uploaded_files = []
    total_size = 0
    for file_path in source_path.rglob('*'):
      if file_path.is_file():
        relative_path = file_path.relative_to(source_path)
        s3_key = f"backups/{backup_name}/{relative_path}"
        with open(file_path, 'rb') as f:
          self.s3_client.upload_fileobj(f, self.bucket_name, s3_key)
        uploaded_files.append(s3_key)
        total_size += file_path.stat().st_size
    return {
      'success': True,
      'location': f"s3://{self.bucket_name}/backups/{backup_name}/",
      'size': total_size,
      'files': uploaded_files
    }

  def _clean_s3_path(self, storage_path: str) -> str:
    if storage_path.startswith('s3://'):
      storage_path = storage_path[5:]
    if storage_path.startswith(f"{self.bucket_name}/"):
      storage_path = storage_path[len(f"{self.bucket_name}/"):]
    return storage_path

  def _delete_s3_directory(self, prefix: str) -> None:
    response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
    if 'Contents' in response:
      objects = [{'Key': obj['Key']} for obj in response['Contents']]
      self.s3_client.delete_objects(Bucket=self.bucket_name, Delete={'Objects': objects})
