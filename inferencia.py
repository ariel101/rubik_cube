from ultralytics import YOLO
import cv2
import time
import numpy as np
from sklearn.cluster import DBSCAN
import kociemba

# ============================= CONFIGURACIÓN =============================
MODEL_PATH = "best150.pt"

BASE_CONF_THRESHOLD = 0.30
RELATIVE_EPS_FACTOR = 1.5
RELATIVE_MIN_SIZE_FACTOR = 0.4
RELATIVE_MAX_SIZE_FACTOR = 2.5
ASPECT_RATIO_MAX = 1.8

MIN_STICKERS_PER_CLUSTER = 9
MAX_STICKERS_PER_CLUSTER = 9
DBSCAN_MIN_SAMPLES = 4

# ============================= MAPA DE COLORES =============================
color_to_letter = {
    'white': 'U',
    'yellow': 'D',
    'red': 'R',
    'orange': 'L',
    'green': 'F',
    'blue': 'B',
}

letter_to_color_name = {v: k.capitalize() for k, v in color_to_letter.items()}

faces_order = ['F', 'U', 'R', 'D', 'L', 'B']

cube_state = {face: None for face in 'URFDLB'}

# ============================= CARGA =============================
model = YOLO(MODEL_PATH)
print("Modelo cargado.")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error al abrir cámara")
    exit()

current_face_index = 0
captured_faces = 0

prev_time = time.time()

print("\n=== SOLVER DE CUBO INTERACTIVO (ESTRICTO: 9 stickers) ===")
print("Ajusta la cámara hasta ver exactamente 9 stickers verdes.")
print("Presiona 'c' para capturar la cara (solo si es correcta).")
print("Presiona 'r' para reiniciar, 'q' para salir.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=BASE_CONF_THRESHOLD, iou=0.45, verbose=False)

    detections = []
    box_sizes = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            if conf < BASE_CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1
            size = (w + h) / 2.0

            if max(w, h) / min(w, h) > ASPECT_RATIO_MAX:
                continue

            center = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
            color_name = model.names[cls_id].lower()

            detections.append({
                'center': center,
                'box': (x1, y1, x2, y2),
                'color_name': color_name,
                'conf': conf,
                'cls': cls_id,
                'size': size
            })
            box_sizes.append(size)

    annotated_frame = frame.copy()

    # Debug rojo tenue: crudas
    for det in detections:
        x1, y1, x2, y2 = det['box']
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

    valid_detections = []
    filtered_detections = []
    avg_box_size = 0.0
    eps_dynamic = 0.0

    potential_grid = None          # Grid temporal de la detección actual
    potential_face_str = None      # String temporal de la cara actual

    if len(detections) > 5 and box_sizes:
        avg_box_size = np.mean(box_sizes)

        filtered_detections = [
            d for d in detections
            if avg_box_size * RELATIVE_MIN_SIZE_FACTOR <= d['size'] <= avg_box_size * RELATIVE_MAX_SIZE_FACTOR
        ]

        for det in filtered_detections:
            x1, y1, x2, y2 = det['box']
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

        if len(filtered_detections) >= MIN_STICKERS_PER_CLUSTER:
            centers = np.array([d['center'] for d in filtered_detections])
            eps_dynamic = avg_box_size * RELATIVE_EPS_FACTOR

            clustering = DBSCAN(eps=eps_dynamic, min_samples=DBSCAN_MIN_SAMPLES).fit(centers)
            labels = clustering.labels_

            unique_labels, counts = np.unique(labels, return_counts=True)

            for label, count in zip(unique_labels, counts):
                if label == -1:
                    continue
                if count == 9:  # Estricto
                    cluster_indices = np.where(labels == label)[0]
                    cluster_dets = [filtered_detections[i] for i in cluster_indices]
                    valid_detections.extend(cluster_dets)

                    mean_center = np.mean(centers[cluster_indices], axis=0).astype(int)
                    cv2.putText(annotated_frame, f"Grupo EXACTO 9", 
                                (mean_center[0], mean_center[1] - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    # Preparar grid
                    sorted_by_y = sorted(cluster_dets, key=lambda d: d['center'][1])
                    row_size = len(sorted_by_y) // 3
                    rows = [sorted_by_y[i:i + row_size] for i in range(0, len(sorted_by_y), row_size)]
                    if len(rows) == 3:
                        potential_grid = []
                        for row in rows:
                            sorted_row = sorted(row, key=lambda d: d['center'][0])
                            potential_grid.extend(sorted_row)

                        if len(potential_grid) == 9:
                            # Dibujar números para verificar orden
                            for i, det in enumerate(potential_grid):
                                cx, cy = map(int, det['center'])
                                cv2.putText(annotated_frame, str(i), (cx - 15, cy + 15),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

                            # Verificar centro y generar string temporal
                            center_sticker = potential_grid[4]
                            center_color = center_sticker['color_name'].lower()
                            expected_color_lower = next(
                                (k.lower() for k, v in color_to_letter.items() if v == faces_order[current_face_index]),
                                None
                            )

                            if expected_color_lower and center_color == expected_color_lower:
                                potential_face_str = ''.join(color_to_letter.get(d['color_name'].lower(), '?') for d in potential_grid)
                                if '?' not in potential_face_str:
                                    cv2.putText(annotated_frame, "Presiona 'c' para capturar (buena detección)",
                                                (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                else:
                                    cv2.putText(annotated_frame, "Error: algún color no mapeado - ajusta",
                                                (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            else:
                                cv2.putText(annotated_frame, "Centro no coincide - ajusta posición",
                                            (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        else:
                            cv2.putText(annotated_frame, "No se formó grid de 9 - ajusta",
                                        (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Dibujar válidas en verde
    for det in valid_detections:
        x1, y1, x2, y2 = det['box']
        label = f"{det['color_name']} {det['conf']:.2f}"
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Instrucción actualizada
    if captured_faces < 6:
        current_face = faces_order[current_face_index]
        expected_color = letter_to_color_name.get(current_face, '?')
        cv2.putText(annotated_frame, f"MUESTRA CARA {current_face} (centro: {expected_color})",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.putText(annotated_frame, f"Capturadas: {captured_faces}/6   -   Presiona 'c' para capturar",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(annotated_frame, "Presiona 'r' para recapturar (reemplaza anterior)",
                    (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    else:
        cv2.putText(annotated_frame, "¡COMPLETADO! Procesando solución...",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 4)

    # Debug
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    debug_text = f"FPS: {fps:.1f} | Crudas: {len(detections)} | Filtradas: {len(filtered_detections)} | Válidas: {len(valid_detections)}"
    if avg_box_size > 0:
        debug_text += f" | Avg size: {avg_box_size:.1f}px | EPS: {eps_dynamic:.1f}"
    cv2.putText(annotated_frame, debug_text, (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Rubik - Clustering Dinámico", annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        cube_state = {face: None for face in 'URFDLB'}
        current_face_index = 0
        captured_faces = 0
        potential_grid = None
        potential_face_str = None
        print("Reiniciado.")
    elif key == ord('c') and potential_face_str:
        current_face = faces_order[current_face_index]
        cube_state[current_face] = potential_face_str
        captured_faces += 1
        current_face_index += 1
        print(f"✓ Cara {current_face} capturada (confirmada): {potential_face_str}")
        potential_grid = None
        potential_face_str = None  # Limpiar para la siguiente cara

    # Resolver al completar
    if captured_faces == 6:
        full_string = ''.join(cube_state[face] for face in 'URFDLB')
        print("\nEstado completo:", full_string)
        try:
            solution = kociemba.solve(full_string)
            print("\n¡SOLUCIÓN!")
            print(solution)
            print(f"Movimientos: {len(solution.split())}")
            cv2.putText(annotated_frame, "SOLUCION: " + solution[:50] + "...",
                        (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
            cv2.imshow("Rubik - Clustering Dinámico", annotated_frame)
            cv2.waitKey(0)  # Pausa para leer
        except Exception as e:
            print("Error:", e)
        break

cap.release()
cv2.destroyAllWindows()