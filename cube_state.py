# cube_state.py

class CubeState:

    def __init__(self, faces_order):
        self.faces = {face: None for face in 'URFDLB'}
        self.faces_order = faces_order
        self.current_index = 0

    def current_face(self):
        return self.faces_order[self.current_index]

    def add_face(self, face_str):
        face = self.current_face()
        self.faces[face] = face_str
        self.current_index += 1

    def is_complete(self):
        return self.current_index == 6

    def build_string(self):
        return ''.join(self.faces[f] for f in 'URFDLB')

    def reset(self):
        self.faces = {face: None for face in 'URFDLB'}
        self.current_index = 0