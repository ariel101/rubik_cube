# grid.py

def build_grid(cluster_dets):

    sorted_by_y = sorted(cluster_dets, key=lambda d: d['center'][1])
    row_size = len(sorted_by_y) // 3

    rows = [
        sorted_by_y[i:i+row_size]
        for i in range(0, len(sorted_by_y), row_size)
    ]

    if len(rows) != 3:
        return None

    grid = []
    for row in rows:
        sorted_row = sorted(row, key=lambda d: d['center'][0])
        grid.extend(sorted_row)

    if len(grid) != 9:
        return None

    return grid