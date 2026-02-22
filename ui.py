# ui.py

import cv2
import time

class UI:

    def __init__(self, window_name="Rubik Solver"):
        self.window_name = window_name
        self.prev_time = time.time()

    def draw_boxes(self, frame, detections, color=(0,255,0)):
        for det in detections:
            x1, y1, x2, y2 = det['box']
            label = f"{det['color_name']} {det['conf']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def draw_text(self, frame, text, pos, scale=0.7, color=(255,255,255), thickness=2):
        cv2.putText(frame, text, pos,
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

    def draw_fps(self, frame):
        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time) if (curr_time - self.prev_time) > 0 else 0
        self.prev_time = curr_time
        self.draw_text(frame, f"FPS: {fps:.1f}", (10, 30))
        return fps

    def show(self, frame):
        cv2.imshow(self.window_name, frame)

    def get_key(self):
        return cv2.waitKey(1) & 0xFF

    def close(self):
        cv2.destroyAllWindows()