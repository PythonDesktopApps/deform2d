from utils import *

class Vertex:
    def __init__(self, pos):
        self.vPosition = vec2(pos)

class Triangle:
    def __init__(self):
        self.nVerts = np.zeros(3, dtype=np.int32)
        self.vTriCoords = np.zeros((3, 2), dtype=np.float64)  # store as double for stability
        self.vScaled = np.zeros((3, 2), dtype=np.float64)
        self.mF = None  # 4x4
        self.mC = None  # 4x6

class Constraint:
    def __init__(self, vidx, pos):
        self.nVertex = int(vidx)
        self.vConstrainedPos = np.array([pos[0], pos[1]], dtype=np.float32)

    def __lt__(self, other):
        return self.nVertex < other.nVertex

    def __eq__(self, other):
        return self.nVertex == other.nVertex

    def __hash__(self):
        return hash(self.nVertex)
