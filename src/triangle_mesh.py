import numpy as np

class TriangleMesh:
    def __init__(self):
        self.vertices = np.zeros((0, 3), dtype=np.float32)
        self.triangles = np.zeros((0, 3), dtype=np.int32)

    def clear(self):
        self.vertices = np.zeros((0, 3), dtype=np.float32)
        self.triangles = np.zeros((0, 3), dtype=np.int32)

    def append_vertex(self, v):
        self.vertices = np.vstack([self.vertices, np.array(v, dtype=np.float32)])

    def append_triangle(self, tri_idx3):
        self.triangles = np.vstack([self.triangles, np.array(tri_idx3, dtype=np.int32)])

    def get_num_vertices(self):
        return self.vertices.shape[0]

    def get_num_triangles(self):
        return self.triangles.shape[0]

    def get_vertex(self, i):
        return self.vertices[i].copy()

    def set_vertex(self, i, v):
        self.vertices[i] = v

    def get_triangle_indices(self, i):
        return self.triangles[i].copy()

    def get_triangle_vertices(self, i):
        idx = self.triangles[i]
        return self.vertices[idx].copy()

    def get_bounding_box(self):
        if self.get_num_vertices() == 0:
            return np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
        mn = self.vertices.min(axis=0)
        mx = self.vertices.max(axis=0)
        return np.array([mn[0], mx[0], mn[1], mx[1], mn[2], mx[2]], dtype=np.float32)

    def read_obj(self, path):
        self.clear()
        verts = []
        faces = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if parts[0] == 'v':
                    if len(parts) >= 4:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    else:
                        x, y = float(parts[1]), float(parts[2])
                        z = 0.0
                    verts.append([x, y, z])
                elif parts[0] == 'f':
                    idxs = []
                    for t in parts[1:4]:
                        s = t.split('/')[0]
                        idxs.append(int(s) - 1)
                    faces.append(idxs)
        if len(verts) == 0 or len(faces) == 0:
            return
        self.vertices = np.array(verts, dtype=np.float32)
        self.triangles = np.array(faces, dtype=np.int32)
