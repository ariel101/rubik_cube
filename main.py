# main.py

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


def main():

    detector = StickerDetector(MODEL_PATH, BASE_CONF_THRESHOLD)
    cube_state = CubeState(faces_order)
    ui = UI()

    cap = cv2.VideoCapture(0)
    capture_face = 0

    while True:

        ret, frame = cap.read()
        if not ret:
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

        if labels is not None:
            unique_labels, counts = np.unique(labels, return_counts=True)

            for label, count in zip(unique_labels, counts):
                if label == -1:
                    continue
                if count == MIN_STICKERS_PER_CLUSTER:
                    indices = np.where(labels == label)[0]
                    valid_cluster = [filtered[i] for i in indices]

        grid = None
        if len(valid_cluster) == 9:
            grid = build_grid(valid_cluster)

        ui.draw_boxes(frame, valid_cluster)
        ui.draw_fps(frame)

        if grid and capture_face < len(faces_order):
            current_face = faces_order[capture_face]
            espected_color = letter_to_color_name[current_face]
            ui.draw_text(frame, "Presiona 'c' para capturar", (10, 70), 0.8, (0,255,0), 2)
            ui.draw_text(frame, f"Capturando cara: {faces_order[capture_face]} (centro: {espected_color})", (10, 110), 0.8, (0,255,0), 2)

        ui.show(frame)

        key = ui.get_key()

        if key == ord('q'):
            break

        if key == ord('r'):
            cube_state.reset()

        if key == ord('c') and grid:
            face_string = ''.join(
                color_to_letter.get(d['color_name'], '?') for d in grid
            )

            if '?' not in face_string and capture_face < len(faces_order):
                cube_state.add_face(face_string)
                capture_face += 1
                print("Cara capturada:", face_string)

        if cube_state.is_complete():
            full_string = cube_state.build_string()
            print("Estado completo:", full_string)
            solution = solve_cube(full_string)
            print("Solución:", solution)
            break

    cap.release()
    ui.close()


if __name__ == "__main__":
    main()