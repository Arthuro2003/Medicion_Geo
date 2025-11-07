from crispy_forms.bootstrap import Field, InlineRadios
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Fieldset, Submit, Row, Column, HTML
from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.core.validators import FileExtensionValidator

from .models import Project, ProjectImage, UserPreferences, VideoSession


class RegisterForm(forms.Form):
  username = forms.CharField(max_length=150, widget=forms.TextInput(attrs={'class': 'form-control'}))
  first_name = forms.CharField(max_length=30, required=False, widget=forms.TextInput(attrs={'class': 'form-control'}))
  last_name = forms.CharField(max_length=150, required=False, widget=forms.TextInput(attrs={'class': 'form-control'}))
  email = forms.EmailField(required=True, widget=forms.EmailInput(attrs={'class': 'form-control'}))
  password = forms.CharField(required=True, widget=forms.PasswordInput(attrs={'class': 'form-control'}))
  confirm_password = forms.CharField(required=True, widget=forms.PasswordInput(attrs={'class': 'form-control'}))

  def clean_username(self):
    username = self.cleaned_data.get('username')
    if User.objects.filter(username__iexact=username).exists():
      raise ValidationError('Ya existe un usuario con ese nombre de usuario.')
    return username

  def clean_email(self):
    email = self.cleaned_data.get('email')
    if User.objects.filter(email__iexact=email).exists():
      raise ValidationError('Ya existe una cuenta registrada con ese correo.')
    return email

  def clean(self):
    cleaned = super().clean()
    pw = cleaned.get('password')
    pw2 = cleaned.get('confirm_password')
    if pw and pw2 and pw != pw2:
      raise ValidationError('Las contraseñas no coinciden.')
    if pw:
      try:
        validate_password(pw)
      except ValidationError as e:
        self.add_error('password', e)
        raise ValidationError(e)
    return cleaned

  def save(self):
    data = self.cleaned_data
    user = User.objects.create_user(
      username=data['username'],
      email=data['email'],
      password=data['password'],
      first_name=data.get('first_name', ''),
      last_name=data.get('last_name', '')
    )
    user.is_staff = False
    user.is_superuser = False
    user.save()
    return user


class MultipleFileInput(forms.ClearableFileInput):
  allow_multiple_selected = True


class MultipleFileField(forms.FileField):
  def __init__(self, *args, **kwargs):
    kwargs.setdefault("widget", MultipleFileInput())
    super().__init__(*args, **kwargs)

  def clean(self, data, initial=None):
    single_file_clean = super().clean
    if isinstance(data, (list, tuple)):
      result = [single_file_clean(d, initial) for d in data]
    else:
      result = single_file_clean(data, initial)
    return result


class ProjectForm(forms.ModelForm):
  class Meta:
    model = Project
    fields = ['name', 'description']
    widgets = {
      'name': forms.TextInput(attrs={
        'class': 'form-control',
        'placeholder': 'Ingrese el nombre del proyecto'
      }),
      'description': forms.Textarea(attrs={
        'class': 'form-control',
        'rows': 4,
        'placeholder': 'Descripción opcional del proyecto'
      })
    }

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.helper = FormHelper()
    self.helper.layout = Layout(
      Fieldset(
        'Información del Proyecto',
        Field('name', css_class='mb-3'),
        Field('description', css_class='mb-3'),
      ),
      Submit('submit', 'Guardar Proyecto', css_class='btn btn-primary')
    )


class ProjectImageForm(forms.ModelForm):
  class Meta:
    model = ProjectImage
    fields = ['name', 'image']
    widgets = {
      'name': forms.TextInput(attrs={
        'class': 'form-control',
        'placeholder': 'Nombre descriptivo para la imagen'
      }),
      'image': forms.FileInput(attrs={
        'class': 'form-control',
        'accept': 'image/*'
      })
    }

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.fields['image'].validators = [
      FileExtensionValidator(allowed_extensions=['jpg', 'jpeg', 'png', 'bmp', 'tiff'])
    ]

    self.helper = FormHelper()
    self.helper.layout = Layout(
      Fieldset(
        'Subir Nueva Imagen',
        Field('name', css_class='mb-3'),
        Field('image', css_class='mb-3'),
        HTML('<small class="form-text text-muted">Formatos soportados: JPG, PNG, BMP, TIFF</small>')
      ),
      Submit('submit', 'Subir Imagen', css_class='btn btn-success')
    )


class ReportGenerationForm(forms.Form):
  REPORT_FORMATS = [
    ('pdf', 'Reporte PDF Completo'),
    ('excel', 'Archivo Excel con Datos'),
    ('json', 'Datos en Formato JSON')
  ]

  format = forms.ChoiceField(
    label='Formato del Reporte',
    choices=REPORT_FORMATS,
    widget=forms.RadioSelect(attrs={'class': 'form-check-input'})
  )

  include_images = forms.BooleanField(
    label='Incluir imágenes en el reporte',
    required=False,
    initial=True,
    widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
  )

  include_measurements = forms.BooleanField(
    label='Incluir mediciones detalladas',
    required=False,
    initial=True,
    widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
  )

  include_shapes = forms.BooleanField(
    label='Incluir formas geométricas',
    required=False,
    initial=True,
    widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
  )

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.helper = FormHelper()
    self.helper.layout = Layout(
      Fieldset(
        'Opciones del Reporte',
        InlineRadios('format'),
        HTML('<hr>'),
        Field('include_images'),
        Field('include_measurements'),
        Field('include_shapes'),
      ),
      Submit('submit', 'Generar Reporte', css_class='btn btn-success')
    )


class ImageFilterForm(forms.Form):
  calibrated = forms.ChoiceField(
    label='Estado de Calibración',
    choices=[
      ('', 'Todas las imágenes'),
      ('yes', 'Solo calibradas'),
      ('no', 'Solo sin calibrar')
    ],
    required=False,
    widget=forms.Select(attrs={'class': 'form-control form-control-sm'})
  )

  has_measurements = forms.ChoiceField(
    label='Con Mediciones',
    choices=[
      ('', 'Todas'),
      ('yes', 'Con mediciones'),
      ('no', 'Sin mediciones')
    ],
    required=False,
    widget=forms.Select(attrs={'class': 'form-control form-control-sm'})
  )

  search = forms.CharField(
    label='Buscar por nombre',
    required=False,
    widget=forms.TextInput(attrs={
      'class': 'form-control form-control-sm',
      'placeholder': 'Buscar imágenes...'
    })
  )

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.helper = FormHelper()
    self.helper.form_method = 'GET'
    self.helper.layout = Layout(
      Row(
        Column('search', css_class='col-md-4'),
        Column('calibrated', css_class='col-md-4'),
        Column('has_measurements', css_class='col-md-4'),
      ),
      Submit('submit', 'Filtrar', css_class='btn btn-outline-primary btn-sm')
    )


class BulkImageUploadForm(forms.Form):
  images = MultipleFileField(
    label='Seleccionar Imágenes',
    widget=MultipleFileInput(attrs={
      'class': 'form-control',
      'multiple': True,
      'accept': 'image/*'
    })
  )

  name_prefix = forms.CharField(
    label='Prefijo para nombres',
    required=False,
    widget=forms.TextInput(attrs={
      'class': 'form-control',
      'placeholder': 'Prefijo opcional para nombres de archivos'
    })
  )

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.helper = FormHelper()
    self.helper.layout = Layout(
      Fieldset(
        'Subida Múltiple de Imágenes',
        Field('images', css_class='mb-3'),
        Field('name_prefix', css_class='mb-3'),
        HTML(
          '<small class="form-text text-muted">Puede seleccionar múltiples imágenes. Formatos soportados: JPG, PNG, BMP, TIFF</small>')
      ),
      Submit('submit', 'Subir Imágenes', css_class='btn btn-success')
    )

  def clean_images(self):
    images = self.files.getlist('images')
    if not images:
      raise forms.ValidationError('Debe seleccionar al menos una imagen.')

    allowed_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
    for image in images:
      ext = image.name.split('.')[-1].lower()
      if ext not in allowed_extensions:
        raise forms.ValidationError(f'Formato no soportado: {image.name}')

    return images


class UserPreferencesForm(forms.ModelForm):
  class Meta:
    model = UserPreferences
    fields = [
      'default_unit', 'theme', 'email_notifications',
      'auto_calibration_suggestions', 'image_quality',
      'auto_save', 'auto_detection', 'auto_backup'
    ]
    widgets = {
      'default_unit': forms.Select(attrs={'class': 'form-select'}),
      'theme': forms.Select(attrs={'class': 'form-select'}),
      'image_quality': forms.Select(attrs={'class': 'form-select'}),
      'email_notifications': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
      'auto_calibration_suggestions': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
      'auto_save': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
      'auto_detection': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
      'auto_backup': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
    }
    labels = {
      'default_unit': 'Unidad de Medida Predeterminada',
      'theme': 'Tema de Interfaz',
      'email_notifications': 'Notificaciones por Email',
      'auto_calibration_suggestions': 'Sugerencias de Calibración',
      'image_quality': 'Calidad de Imagen',
      'auto_save': 'Auto-guardado',
      'auto_detection': 'Detección Automática',
      'auto_backup': 'Respaldo Automático',
    }


class VideoSessionForm(forms.ModelForm):
  class Meta:
    model = VideoSession
    fields = ['name']
    widgets = {
      'name': forms.TextInput(attrs={
        'class': 'form-control',
        'placeholder': 'Nombre de la sesión de video'
      })
    }
    labels = {
      'name': 'Nombre de la Sesión'
    }

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.helper = FormHelper()
    self.helper.layout = Layout(
      Fieldset(
        'Nueva Sesión de Video',
        Field('name', css_class='mb-3'),
        HTML('<small class="form-text text-muted">Se utilizará la cámara principal del sistema.</small>')
      ),
      Submit('submit', 'Iniciar Sesión de Video', css_class='btn btn-primary')
    )


class VideoCalibrationForm(forms.Form):
  UNIT_CHOICES = [
    ('mm', 'Milímetros (mm)'),
    ('cm', 'Centímetros (cm)'),
    ('m', 'Metros (m)'),
    ('in', 'Pulgadas (in)'),
    ('ft', 'Pies (ft)'),
  ]

  real_distance = forms.FloatField(
    label='Distancia Real',
    min_value=0.001,
    widget=forms.NumberInput(attrs={
      'class': 'form-control',
      'placeholder': 'Ingrese la distancia real conocida',
      'step': '0.001'
    })
  )

  unit = forms.ChoiceField(
    label='Unidad de Medida',
    choices=UNIT_CHOICES,
    widget=forms.Select(attrs={'class': 'form-control'})
  )

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.helper = FormHelper()
    self.helper.layout = Layout(
      Fieldset(
        'Calibración de Video',
        HTML(
          '<div class="alert alert-info">Seleccione dos puntos en el video y proporcione la distancia real entre ellos.</div>'),
        Row(
          Column('real_distance', css_class='col-md-6'),
          Column('unit', css_class='col-md-6'),
        )
      ),
      Submit('submit', 'Calibrar Video', css_class='btn btn-primary')
    )
