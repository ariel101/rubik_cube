# processing.py

import numpy as np

def extract_detections(results, model, aspect_ratio_max, conf_threshold):
    detections = []
    box_sizes = []

    for result in results:
        for box in result.boxes:

            conf = float(box.conf)
            if conf < conf_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1

            if max(w, h) / min(w, h) > aspect_ratio_max:
                continue

            center = [(x1 + x2)/2, (y1 + y2)/2]
            size = (w + h) / 2.0
            color_name = model.names[int(box.cls)].lower()

            detections.append({
                'center': center,
                'box': (x1, y1, x2, y2),
                'color_name': color_name,
                'conf': conf,
                'size': size
            })

            box_sizes.append(size)

    return detections, box_sizes


def filter_by_size(detections, box_sizes, min_factor, max_factor):

    if not box_sizes:
        return [], 0

    avg_size = np.mean(box_sizes)

    filtered = [
        d for d in detections
        if avg_size * min_factor <= d['size'] <= avg_size * max_factor
    ]

    return filtered, avg_size