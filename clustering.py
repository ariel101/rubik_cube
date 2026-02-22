# clustering.py

import numpy as np
from sklearn.cluster import DBSCAN

def cluster_stickers(filtered_detections, avg_size, eps_factor, min_samples):

    if len(filtered_detections) < 9:
        return None

    centers = np.array([d['center'] for d in filtered_detections])
    eps_dynamic = avg_size * eps_factor

    clustering = DBSCAN(
        eps=eps_dynamic,
        min_samples=min_samples
    ).fit(centers)

    return clustering.labels_