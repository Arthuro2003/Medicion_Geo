---

# MedicionGeo

MedicionGeo es una aplicación web para **medir distancias y dimensiones** en tiempo real usando la **cámara del dispositivo** y a partir de **imágenes** cargadas. Integra un flujo de proyectos, galería, detalle de imágenes con lienzo de medición, y sesiones de video con calibración (incluyendo soporte para marcadores ArUco).

---

## ✨ Características

* **Proyectos**: creación, listado, detalle, exportaciones (PDF/JSON) y eliminación con doble confirmación.
* **Video en tiempo real**: vista previa del stream, selección de cámara por índice, captura de clics, medición y calibración en vivo.
* **Imágenes**: carga múltiple, metadatos, estados “Calibrada/Sin calibrar”, mediciones sobre `<canvas>`, detección automática (ArUco).
* **Galería**: filtros rápidos, `loading="lazy"` para rendimiento y acciones contextuales.
* **Estadísticas & Dashboard**: KPIs resumidos (calibración, actividad, proyectos más activos).
* **Perfil & Preferencias**: unidades, tema (claro/oscuro/auto), autosave, backups, opciones de detección.
* **Accesibilidad y UX**: mensajes con `aria-live`, botones con `aria-label`, estados consistentes.
* **Seguridad en frontend y backend**:

  * CSRF en formularios y `fetch`.
  * Verificación estricta de propiedad (`created_by=request.user`) en vistas sensibles.
  * Confirmaciones reforzadas en eliminaciones.
  * Sanitización y uso de `textContent` para mensajes dinámicos.

---

## 🧱 Arquitectura (alto nivel)

* **Django 4.2** (LTS) con vistas clásicas + endpoints JSON.
* **Django Channels** (opcional) y Redis (opcional) para actualizaciones en tiempo real.
* **Frontend**: plantillas HTML Bootstrap + JS ligero (sin build step).
* **Visión por Computador**: OpenCV / MediaPipe (opcional según features usadas).
* **Almacenamiento**: por defecto SQLite + `MEDIA_ROOT` para imágenes.

> Nota: algunas vistas de video usan handlers en memoria para el stream. En despliegues concurrentes se recomienda un “handler por usuario/sesión” o mover el procesamiento a workers (Channels/WebSocket).

---

## 📁 Estructura relevante

```
core/                   # app principal
  admin.py
  apps.py
  context_processors.py
  forms.py
  models.py
  tests.py
  urls.py
  views.py
  views_video.py

templates/
  base.html
  dashboard.html
  video_dashboard.html / start_video_session.html
  project_list.html / project_detail.html / project_form.html
  project_confirm_delete.html
  gallery.html / image_detail.html / image_confirm_delete.html
  statistics.html / user_profile.html / user_settings.html
  login.html / register.html
  help.html

static/                 # (si aplica)
media/                  # subidas de imágenes
```

---

⚠️ Importante !!! 

Para poder utilizar nuestro proyecto es de vital importancia que tengas disponible impreso un aruco de las medidad 5cm x 5cm. Puedes descargarlo desde la carpeta "aruco_descargable_5cmx5cm" o descagalo directamente atraves del siguiente imagen:

<img width="200" height="200" alt="imagen" src="https://github.com/user-attachments/assets/f59f4425-8db2-4689-952e-3d3f887f8a6c" />

---

## 🚀 Empezar (local)

### 1) Requisitos del sistema

* **Python 3.10+** (recomendado 3.10/3.11)
* **pip** y **venv**
* Si usarás streaming/detección avanzada:

  * **OpenCV** (se instala vía `pip`, pero en Linux quizá requieras paquetes del SO).
  * **Redis** si activarás Channels (opcional).

### 2) Clonar y crear entorno

```bash
git clone https://github.com/Arthuro2003/Medicion_Geo.git
cd <tu-repo>

python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
```

### 3) Dependencias

Instala los requisitos base:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> Sugerido: separar dependencias pesadas de ML/visión en un archivo extra (`requirements-ml.txt`) si no usarás todo en desarrollo.

### 4) Variables de entorno

Crea un archivo `.env` en la raíz del proyecto:

```
DEBUG=True
SECRET_KEY=pon_aqui_una_llave_segura
ALLOWED_HOSTS=localhost,127.0.0.1
CSRF_TRUSTED_ORIGINS=http://localhost,http://127.0.0.1

# Ruta de media (si no se define, usarás lo de settings)
MEDIA_ROOT=./media
```

> En producción, desactiva `DEBUG`, usa `SECRET_KEY` robusta y configura HTTPS.

### 5) Migraciones y usuario admin

```bash
python manage.py migrate
python manage.py createsuperuser
```

### 6) Ejecutar servidor

```bash
python manage.py runserver
```

Abre: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

---

## 🔌 Integración opcional: Redis + Channels

Si vas a usar características en tiempo real:

```bash
# con Docker
docker run -p 6379:6379 --name redis -d redis:7-alpine
```

Configura en `settings.py` (ejemplo):

```python
ASGI_APPLICATION = "project.asgi.application"
CHANNEL_LAYERS = {
  "default": {
    "BACKEND": "channels_redis.core.RedisChannelLayer",
    "CONFIG": {"hosts": [("127.0.0.1", 6379)]},
  }
}
```

Arranca el servidor ASGI (si aplica):

```bash
daphne project.asgi:application
# o uvicorn si lo usas
```

---

## 📷 Cámara y permisos (guía rápida)

* **HTTPS**: los navegadores piden sitio seguro para acceso a cámara. En local, `http://localhost` está permitido; en despliegues, usa **HTTPS**.
* **Permisos**: si no ves video, revisa:

  1. Permiso de cámara concedido al sitio.
  2. Dispositivo: selecciona el índice correcto (al cambiar el índice, reinicia la vista previa).
  3. Cierra otras apps que estén usando la cámara.
* **Errores legibles**: el front muestra alertas accesibles con `aria-live`.

---

## 🧭 Flujo principal de uso

1. **Crea un proyecto**: nombre y descripción.
2. **Inicia sesión de video**: selecciona cámara, calibra (p. ej., con una distancia conocida o marcador ArUco), toma mediciones.
3. **Carga imágenes**: desde el proyecto, sube una o varias. Puedes:

   * Calibrar por referencia (píxeles → unidad).
   * Medir sobre el `<canvas>`.
   * Ejecutar detección automática (si activada).
4. **Galería**: filtra por calibradas/sin calibrar y entra al detalle para medir.
5. **Exporta**: genera reportes o descarga datos.

---

## 🔐 Notas de seguridad (resumen aplicado)

* **CSRF**: formularios y `fetch` usan token; en endpoints JSON preferimos mantener CSRF habilitado.
* **Propiedad/Permisos**: todas las vistas sensibles deben obtener objetos con `project__created_by=request.user` (o equivalente) para evitar accesos ajenos.
* **Mensajes en el DOM**: usar `textContent` para texto variable; evitar `innerHTML` con datos no confiables.
* **Eliminaciones**: doble confirmación (checkbox + confirm + prompt “ELIMINAR”) y feedback visual con spinner.

---

## 🧩 Endpoints y URLs

* **Proyectos**: listado, detalle, crear/editar, eliminar.
* **Video**:

  * `projects/<uuid:project_pk>/video/<uuid:session_pk>/` — stream/control de una sesión.
  * `measurement/stream/` — stream genérico (según configuración).
* **Imágenes**: galería, detalle `<canvas>`, detección (`POST`), eliminar.
* **Utilidades**: exportación de datos, chequeo de ArUco, APIs JSON.

> Revisa `core/urls.py` para el mapeo completo y nombres de ruta.

---

## 🧪 Pruebas

Ejecución básica:

```bash
python manage.py test
```

Sugerencias de pruebas mínimas:

* Acceso a vistas de video con proyecto de otro usuario → 404.
* Rutas sin colisiones (`video_stream` con kwargs, `measurement_stream` sin kwargs).
* Transaccionalidad en detección: si falla a mitad, no quedan datos a medias.

---

## 🛠️ Mantenimiento y calidad

* **Requisitos**:

  * Evita duplicados: usa **solo** `opencv-contrib-python` (no mezclar con `opencv-python`).
  * Alinea `boto3` y `botocore` a la misma serie de versión.
  * Considera dividir:

    * `requirements.txt` (core)
    * `requirements-ml.txt` (visión/torch)
    * `requirements-dev.txt` (pytest, black, isort, django-debug-toolbar)
* **Accesibilidad**:

  * `aria-label` en icon buttons.
  * `role="status"` y `aria-live` para mensajes de guardado/errores.
* **Rendimiento**:

  * Evita pings pesados a endpoints de streaming. Usa un endpoint ligero de estado o SSE/WebSocket.
* **Frontend DRY**:

  * Extrae lógica repetida (tema, confirm-delete, highlight) a JS compartido.
  * Unifica la librería de íconos en todas las plantillas.

---

## 🧰 Solución de problemas

**No funciona la cámara**

* Verifica permisos del navegador.
* Usa HTTPS en despliegue.
* Revisa que no haya otra app usando la cámara.
* Cambia el índice de cámara y reinicia la vista previa.

**Las fechas “Recientes” no ordenan bien**

* Usa atributos `data-created="YYYY-MM-DD"` para evitar parseos ambiguos por locale.

**PDF con nombre extraño**

* Normaliza el nombre del archivo (solo `A-Za-z0-9_.-`) antes de enviarlo en `Content-Disposition`.

---

## 📜 Licencia

Este proyecto puede distribuirse bajo licencia **MIT** (o la que definas).
Incluye un archivo `LICENSE` en la raíz del repositorio si aún no existe.

---

## 🤝 Contribuir

1. Crea una rama `feature/nombre-feature`.
2. Sigue la guía de estilo (pep8/black).
3. Acompaña cambios de pruebas cuando aplique.
4. Abre un PR describiendo:

   * Cambios de seguridad (si los hay).
   * Migraciones necesarias.
   * Impacto en UX/Accesibilidad.

---

## 🗺️ Roadmap sugerido

* WebSocket/SSE para estado de sesión y métricas en vivo.
* `VideoStreamHandler` por usuario/sesión (evitar estado global).
* Módulos front compartidos: `theme.js`, `confirm-delete.js`, `highlight.js`.
* Detector configurable (tamaño ArUco desde UI, logs de inferencia).
* Exportaciones incrementales y paginadas para grandes volúmenes.

---

**Hecho con ❤️ para mediciones precisas en contextos reales (aula, laboratorio y campo).**
