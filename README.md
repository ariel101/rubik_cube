# 🧩 Rubik Cube Solver

Aplicación web que utiliza **visión por computadora** e **Inteligencia Artificial** para detectar los colores de un cubo de Rubik en tiempo real mediante una cámara web. Una vez capturadas las seis caras del cubo, el sistema genera automáticamente la secuencia de movimientos necesaria para resolverlo utilizando el **algoritmo de Kociemba**.

---

# 📖 Descripción

El proyecto permite resolver un cubo de Rubik de manera automática siguiendo el siguiente proceso:

1. Captura video en tiempo real desde una cámara web.
2. Detecta cada pegatina (sticker) del cubo utilizando un modelo **YOLOv8n** entrenado.
3. Identifica los colores de cada sticker.
4. Agrupa automáticamente las pegatinas pertenecientes a una misma cara mediante **DBSCAN**.
5. Ordena las pegatinas en una cuadrícula **3×3**.
6. El usuario captura las seis caras del cubo.
7. El sistema construye el estado completo del cubo.
8. El algoritmo de **Kociemba** calcula la solución óptima.
9. La secuencia de movimientos se muestra en la interfaz web.

---

# 🚀 Características

- Detección de colores en tiempo real.
- Streaming de video desde la cámara.
- Captura guiada de las seis caras del cubo.
- Agrupamiento automático de stickers mediante DBSCAN.
- Construcción automática del estado del cubo.
- Resolución mediante el algoritmo de Kociemba.
- Interfaz web sencilla e intuitiva.
- Reinicio de captura sin necesidad de reiniciar el servidor.

---

# 🛠️ Tecnologías utilizadas

## Backend

- Python
- FastAPI

## Frontend

- HTML
- CSS
- JavaScript

## Gestor de paquetes

- **Pixi** → Gestión del entorno de desarrollo y las dependencias del proyecto.

## Librerías principales

- **Ultralytics (YOLOv8)** → Detección de stickers y colores.
- **OpenCV** → Captura y procesamiento de video.
- **Scikit-learn** → Agrupamiento mediante DBSCAN.
- **Kociemba** → Resolución del cubo de Rubik.

---

# 📂 Estructura del proyecto

```text
rubik_cube/
│
├── main.py              # API FastAPI y streaming de video
├── detector.py          # Carga e inferencia del modelo YOLO
├── processing.py        # Filtrado y procesamiento de detecciones
├── clustering.py        # Agrupamiento de stickers con DBSCAN
├── grid.py              # Organización de stickers en una matriz 3x3
├── cube_state.py        # Manejo del estado de las seis caras
├── solver.py            # Comunicación con el algoritmo de Kociemba
├── ui.py                # Dibujado de cajas, texto e información en pantalla
├── config.py            # Configuración general del sistema
│
├── best150.pt           # Modelo YOLO entrenado
│
├── templates/
│   └── index.html       # Interfaz principal
│
└── static/
    ├── app.js           # Comunicación con la API
    └── style.css        # Estilos de la aplicación
```

---

# ⚙️ Funcionamiento del sistema

El flujo completo del sistema es el siguiente:

```text
Webcam
   │
   ▼
Captura de video
   │
   ▼
YOLOv8 detecta los stickers
   │
   ▼
Filtrado de detecciones
   │
   ▼
DBSCAN agrupa los stickers
   │
   ▼
Construcción de una cuadrícula 3×3
   │
   ▼
Captura de las seis caras
   │
   ▼
Construcción del estado del cubo
   │
   ▼
Algoritmo de Kociemba
   │
   ▼
Secuencia de movimientos
```

---

# 🎯 Modelo de detección

El sistema utiliza un modelo **YOLOv8n** entrenado para detectar e identificar los colores de las pegatinas del cubo de Rubik.

Para el entrenamiento del modelo se utilizó el siguiente conjunto de datos de **Roboflow Universe**:

**Dataset:** RubyRizz

https://universe.roboflow.com/main-d3i3y/rubyrizz

A partir de este dataset se entrenó el modelo **`best150.pt`**, el cual es utilizado por la aplicación para realizar la detección de los stickers del cubo en tiempo real.

---

# 📸 Flujo de uso

1. Ejecutar la aplicación.
2. Abrir el navegador en:

```
http://localhost:8000
```

3. Mostrar una cara del cubo frente a la cámara.
4. Esperar a que el sistema detecte correctamente los nueve stickers.
5. Presionar **Capturar Cara**.
6. Repetir el proceso hasta capturar las seis caras del cubo.
7. Una vez registradas todas las caras, el sistema calculará automáticamente la solución.
8. La secuencia de movimientos aparecerá en la interfaz web.
9. Para comenzar nuevamente, presionar **Reiniciar**.

---

# ▶️ Instalación

## Clonar el repositorio

```bash
git clone https://github.com/ariel101/rubik_cube.git

cd rubik_cube
```

## Instalar Pixi

Instala **Pixi** siguiendo las instrucciones oficiales para tu sistema operativo:

https://pixi.sh/latest/

## Instalar las dependencias

```bash
pixi install
```

---

# ▶️ Ejecutar el proyecto

Si el proyecto define una tarea para iniciar la aplicación:

```bash
pixi run start
```

O ejecutar directamente el servidor:

```bash
pixi run uvicorn main:app --reload
```

Luego abrir en el navegador:

```
http://localhost:8000
```

---

# 📋 Requisitos

- Python 3.10 o superior.
- Pixi.
- Webcam conectada.
- Cubo de Rubik 3×3 con colores estándar:
  - Blanco
  - Amarillo
  - Rojo
  - Naranja
  - Verde
  - Azul
- Buena iluminación para una mejor detección.

---

# 🧠 Algoritmos utilizados

## YOLOv8

Modelo de detección de objetos encargado de identificar cada sticker del cubo y clasificar su color en tiempo real.

## DBSCAN

Algoritmo de clustering utilizado para agrupar automáticamente las detecciones pertenecientes a una misma cara del cubo.

## Ordenamiento espacial

Una vez agrupadas las detecciones, estas se organizan en una cuadrícula **3×3**, permitiendo reconstruir correctamente la distribución de colores de cada cara.

## Algoritmo de Kociemba

Con las seis caras capturadas, el algoritmo genera una solución eficiente para resolver el cubo utilizando una cantidad reducida de movimientos.

---

# 📷 Interfaz

La aplicación web cuenta con:

- Vista en tiempo real de la cámara.
- Detección de stickers sobre el video.
- Indicador de la cara que debe capturarse.
- Botón para capturar cada cara.
- Botón para reiniciar el proceso.
- Estado de las caras registradas.
- Visualización automática de la secuencia de movimientos para resolver el cubo.

---

# 📌 Futuras mejoras

- Soporte para cubos 2×2 y 4×4.
- Visualización 3D del cubo.
- Animación paso a paso de la solución.
- Optimización del modelo para dispositivos de bajos recursos.
- Mejora de la detección bajo diferentes condiciones de iluminación.
- Exportación de la solución a formatos de texto o imagen.

---

# 👨‍💻 Autor

Proyecto desarrollado como una aplicación de **Visión por Computadora** para la detección y resolución automática de un cubo de Rubik utilizando **YOLOv8**, **OpenCV**, **FastAPI**, **DBSCAN** y el algoritmo de **Kociemba**.