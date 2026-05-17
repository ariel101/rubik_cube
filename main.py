from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

import cv2
import numpy as np

from config import *
from detector import StickerDetector
from processing import extract_detections, filter_by_size
from clustering import cluster_stickers
from grid import build_grid
from cube_state import CubeState
from solver import solve_cube
from ui import UI

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# =========================
# VARIABLES GLOBALES
# =========================

detector = StickerDetector(MODEL_PATH, BASE_CONF_THRESHOLD)
cube_state = CubeState(faces_order)
ui = UI()

cap = cv2.VideoCapture(0)

capture_face = 0
last_grid = None
last_center_color = "?"
expected_color = "?"
solution_text = ""

# =========================
# VIDEO STREAM
# =========================

def generate_frames():

    global capture_face
    global last_grid
    global last_center_color
    global expected_color
    global solution_text

    while True:

        success, frame = cap.read()

        if not success:
            break

        results = detector.detect(frame)

        detections, box_sizes = extract_detections(
            results,
            detector.model,
            ASPECT_RATIO_MAX,
            BASE_CONF_THRESHOLD
        )

        filtered, avg_size = filter_by_size(
            detections,
            box_sizes,
            RELATIVE_MIN_SIZE_FACTOR,
            RELATIVE_MAX_SIZE_FACTOR
        )

        labels = cluster_stickers(
            filtered,
            avg_size,
            RELATIVE_EPS_FACTOR,
            DBSCAN_MIN_SAMPLES
        )

        valid_cluster = []
        grid = None

        if labels is not None:

            unique_labels, counts = np.unique(labels, return_counts=True)

            for label, count in zip(unique_labels, counts):

                if label == -1:
                    continue

                if count == MIN_STICKERS_PER_CLUSTER:

                    indices = np.where(labels == label)[0]
                    valid_cluster = [filtered[i] for i in indices]

        # GRID
        if len(valid_cluster) == 9:
            grid = build_grid(valid_cluster)
            last_grid = grid

        # DIBUJAR CAJAS
        ui.draw_boxes(frame, valid_cluster)
        ui.draw_fps(frame)

        # TEXTO
        if grid and capture_face < len(faces_order):

            center_sticker = grid[4]

            center_color_name = center_sticker['color_name']
            last_center_color = center_color_name

            expected_color = letter_to_color_name[
                faces_order[capture_face]
            ]

            cv2.putText(
                frame,
                f"Detectado: {center_color_name}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )

            cv2.putText(
                frame,
                f"Esperado: {expected_color}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,255),
                2
            )

        # ENCODE FRAME
        _, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )

# =========================
# ROUTES
# =========================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request
        }
    )

@app.get("/video")
def video_feed():

    return StreamingResponse(
        generate_frames(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

@app.post("/capture")
def capture():

    global capture_face
    global solution_text

    if last_grid is None:
        return JSONResponse({
            "status": "error"
        })

    face_string = ''.join(
        color_to_letter.get(d['color_name'], '?')
        for d in last_grid
    )

    if '?' not in face_string:

        cube_state.add_face(face_string)
        capture_face += 1

    if cube_state.is_complete():

        full_string = cube_state.build_string()
        solution_text = solve_cube(full_string)

    return JSONResponse({
        "status": "ok",
        "faces": capture_face,
        "solution": solution_text
    })

@app.post("/reset")
def reset():

    global capture_face
    global solution_text

    cube_state.reset()

    capture_face = 0
    solution_text = ""

    return JSONResponse({
        "status": "reset"
    })

@app.get("/status")
def status():

    current_face = (
        faces_order[capture_face]
        if capture_face < len(faces_order)
        else "COMPLETO"
    )

    return {
        "current_face": current_face,
        "detected": last_center_color,
        "expected": expected_color,
        "solution": solution_text
    }