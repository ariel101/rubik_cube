# config.py

MODEL_PATH = "best150.pt"

BASE_CONF_THRESHOLD = 0.30
RELATIVE_EPS_FACTOR = 1.5
RELATIVE_MIN_SIZE_FACTOR = 0.4
RELATIVE_MAX_SIZE_FACTOR = 2.5
ASPECT_RATIO_MAX = 1.8

MIN_STICKERS_PER_CLUSTER = 9
DBSCAN_MIN_SAMPLES = 4

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