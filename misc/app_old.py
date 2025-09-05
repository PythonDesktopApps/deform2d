import sys
import math
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QApplication
from PyQt5.QtOpenGL import QGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *

# ---------------- Mesh ----------------

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


# --------------- Deformer (full port) ---------------

def vec2(v):
    return np.array(v, dtype=np.float32).reshape(2,)

def length2(v):
    return float(np.linalg.norm(v))

def normalize2(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v

def perp(v):
    return np.array([v[1], -v[0]], dtype=np.float32)

def barycentric_coords(p, a, b, c):
    # 2D barycentric
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-20:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.array([u, v, w], dtype=np.float32)

class RigidMeshDeformer2D:
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

    def __init__(self):
        self.m_vConstraints = set()
        self.m_vInitialVerts = []
        self.m_vDeformedVerts = []
        self.m_vTriangles = []
        self.m_bSetupValid = False

        # Precomputed matrices/structures
        self.m_vVertexMap = []      # size N
        self.m_mFirstMatrix = None  # (2*free, 2*constraints) final matrix used for main solve
        self.mHXPrime = None
        self.mHYPrime = None
        self.mDX = None
        self.mDY = None
        self.mLUDecompX = None
        self.mLUDecompY = None

    def InvalidateSetup(self):
        self.m_bSetupValid = False

    def SetDeformedHandle(self, nHandle, vHandle):
        c = self.Constraint(nHandle, vHandle)
        self.UpdateConstraint(c)

    def RemoveHandle(self, nHandle):
        c = self.Constraint(nHandle, np.array([0.0, 0.0], dtype=np.float32))
        if c in self.m_vConstraints:
            self.m_vConstraints.remove(c)
        # restore to initial
        self.m_vDeformedVerts[nHandle].vPosition = self.m_vInitialVerts[nHandle].vPosition.copy()
        self.InvalidateSetup()

    def UnTransformPoint(self, vTransform):
        # Convert current deformed-space point into initial-space via barycentrics
        for t in self.m_vTriangles:
            v1 = self.m_vDeformedVerts[t.nVerts[0]].vPosition
            v2 = self.m_vDeformedVerts[t.nVerts[1]].vPosition
            v3 = self.m_vDeformedVerts[t.nVerts[2]].vPosition
            b = barycentric_coords(vec2(vTransform), v1, v2, v3)
            if np.all(b >= 0) and np.all(b <= 1):
                v1i = self.m_vInitialVerts[t.nVerts[0]].vPosition
                v2i = self.m_vInitialVerts[t.nVerts[1]].vPosition
                v3i = self.m_vInitialVerts[t.nVerts[2]].vPosition
                v = b[0] * v1i + b[1] * v2i + b[2] * v3i
                vTransform[:] = v
                return

    def InitializeFromMesh(self, mesh: TriangleMesh):
        self.m_vConstraints.clear()
        self.m_vInitialVerts = []
        self.m_vDeformedVerts = []
        self.m_vTriangles = []

        # copy vertices
        nVerts = mesh.get_num_vertices()
        for i in range(nVerts):
            v3 = mesh.get_vertex(i)
            v = self.Vertex([v3[0], v3[1]])
            self.m_vInitialVerts.append(self.Vertex(v.vPosition))
            self.m_vDeformedVerts.append(self.Vertex(v.vPosition))

        # copy triangles
        nTris = mesh.get_num_triangles()
        for i in range(nTris):
            t = RigidMeshDeformer2D.Triangle()
            idx = mesh.get_triangle_indices(i)
            t.nVerts[:] = idx
            self.m_vTriangles.append(t)

        # triangle-local coordinates
        for t in self.m_vTriangles:
            for j in range(3):
                n0 = j
                n1 = (j + 1) % 3
                n2 = (j + 2) % 3
                v0 = self.m_vInitialVerts[t.nVerts[n0]].vPosition.astype(np.float64)
                v1 = self.m_vInitialVerts[t.nVerts[n1]].vPosition.astype(np.float64)
                v2 = self.m_vInitialVerts[t.nVerts[n2]].vPosition.astype(np.float64)
                v01 = v1 - v0
                v01_perp = np.array([v01[1], -v01[0]], dtype=np.float64)
                # project v2 in basis (v01, perp(v01)) with non-orthonormal scaling
                vLocal = v2 - v0
                x = float(np.dot(vLocal, v01) / max(1e-20, np.dot(v01, v01)))
                y = float(np.dot(vLocal, v01_perp) / max(1e-20, np.dot(v01_perp, v01_perp)))
                # sanity is skipped (no DebugBreak)
                t.vTriCoords[j] = [x, y]

        self.InvalidateSetup()

    def UpdateDeformedMesh(self, mesh: TriangleMesh, bRigid=True):
        self.ValidateDeformedMesh(bRigid)
        vVerts = self.m_vDeformedVerts if len(self.m_vConstraints) > 1 else self.m_vInitialVerts
        nVerts = mesh.get_num_vertices()
        arr = mesh.vertices.copy()
        for i in range(nVerts):
            p = vVerts[i].vPosition
            arr[i, 0] = p[0]
            arr[i, 1] = p[1]
            arr[i, 2] = 0.0
        mesh.vertices = arr

    def UpdateConstraint(self, cons):
        if cons in self.m_vConstraints:
            # update existing
            # set position directly
            for c in list(self.m_vConstraints):
                if c == cons:
                    c.vConstrainedPos = cons.vConstrainedPos.copy()
                    break
            self.m_vDeformedVerts[cons.nVertex].vPosition = cons.vConstrainedPos.copy()
        else:
            self.m_vConstraints.add(cons)
            self.m_vDeformedVerts[cons.nVertex].vPosition = cons.vConstrainedPos.copy()
            self.InvalidateSetup()

    def ExtractSubMatrix(self, M, rowOffset, colOffset, rows, cols):
        return M[rowOffset:rowOffset+rows, colOffset:colOffset+cols].copy()

    def ValidateSetup(self):
        if self.m_bSetupValid or len(self.m_vConstraints) < 2:
            return

        self.PrecomputeOrientationMatrix()
        for i in range(len(self.m_vTriangles)):
            self.PrecomputeScalingMatrices(i)
        self.PrecomputeFittingMatrices()

        self.m_bSetupValid = True

    def PrecomputeFittingMatrices(self):
        # constraints as sorted vector
        constraints_vec = sorted(list(self.m_vConstraints), key=lambda c: c.nVertex)

        nVerts = len(self.m_vDeformedVerts)
        nConstraints = len(constraints_vec)
        nFreeVerts = nVerts - nConstraints

        # vertex map: free first, then constraints
        self.m_vVertexMap = [-1] * nVerts
        row = 0
        for i in range(nVerts):
            c = self.Constraint(i, [0.0, 0.0])
            if c in self.m_vConstraints:
                continue
            self.m_vVertexMap[i] = row
            row += 1
        assert row == nFreeVerts
        for i in range(nConstraints):
            self.m_vVertexMap[constraints_vec[i].nVertex] = row
            row += 1
        assert row == nVerts

        # HX and HY (nVerts x nVerts)
        HX = np.zeros((nVerts, nVerts), dtype=np.float64)
        HY = np.zeros((nVerts, nVerts), dtype=np.float64)

        for t in self.m_vTriangles:
            for j in range(3):
                nA = self.m_vVertexMap[t.nVerts[j]]
                nB = self.m_vVertexMap[t.nVerts[(j + 1) % 3]]

                HX[nA, nA] += 2.0
                HX[nA, nB] += -2.0
                HX[nB, nA] += -2.0
                HX[nB, nB] += 2.0

                HY[nA, nA] += 2.0
                HY[nA, nB] += -2.0
                HY[nB, nA] += -2.0
                HY[nB, nB] += 2.0

        # submatrices
        HX00 = self.ExtractSubMatrix(HX, 0, 0, nFreeVerts, nFreeVerts)
        HY00 = self.ExtractSubMatrix(HY, 0, 0, nFreeVerts, nFreeVerts)

        HX01 = self.ExtractSubMatrix(HX, 0, nFreeVerts, nFreeVerts, nConstraints)
        HX10 = self.ExtractSubMatrix(HX, nFreeVerts, 0, nConstraints, nFreeVerts)

        HY01 = self.ExtractSubMatrix(HY, 0, nFreeVerts, nFreeVerts, nConstraints)
        HY10 = self.ExtractSubMatrix(HY, nFreeVerts, 0, nConstraints, nFreeVerts)

        # primes and D
        self.mHXPrime = HX00.copy()  # originally HX00 + HX00^T; original code later replaced with HX00 (same as Swift)
        self.mHYPrime = HY00.copy()

        self.mDX = HX01.copy()  # originally HX01 + HX10^T; original C++ comments had sums but then used HX01 only
        self.mDY = HY01.copy()

        # LU decompositions (store factors via numpy.linalg.lu not exposed; we’ll store inverses for simplicity)
        # For stability and speed you could use scipy.linalg.lu_factor/lu_solve.
        # Here we keep it simple with inversion once per setup:
        try:
            self.mLUDecompX = np.linalg.inv(self.mHXPrime)
            self.mLUDecompY = np.linalg.inv(self.mHYPrime)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            self.mLUDecompX = np.linalg.pinv(self.mHXPrime)
            self.mLUDecompY = np.linalg.pinv(self.mHYPrime)

    def PrecomputeOrientationMatrix(self):
        constraints_vec = sorted(list(self.m_vConstraints), key=lambda c: c.nVertex)
        nVerts = len(self.m_vDeformedVerts)
        nConstraints = len(constraints_vec)
        nFreeVerts = nVerts - nConstraints

        # vertex map: free first, then constraints
        self.m_vVertexMap = [-1] * nVerts
        row = 0
        for i in range(nVerts):
            c = self.Constraint(i, [0.0, 0.0])
            if c in self.m_vConstraints:
                continue
            self.m_vVertexMap[i] = row
            row += 1
        assert row == nFreeVerts
        for i in range(nConstraints):
            self.m_vVertexMap[constraints_vec[i].nVertex] = row
            row += 1
        assert row == nVerts

        size = 2 * nVerts
        First = np.zeros((size, size), dtype=np.float64)

        # fill matrix
        for t in self.m_vTriangles:
            for j in range(3):
                n0x = 2 * self.m_vVertexMap[t.nVerts[j]]
                n0y = n0x + 1
                n1x = 2 * self.m_vVertexMap[t.nVerts[(j + 1) % 3]]
                n1y = n1x + 1
                n2x = 2 * self.m_vVertexMap[t.nVerts[(j + 2) % 3]]
                n2y = n2x + 1
                x = float(t.vTriCoords[j][0])
                y = float(t.vTriCoords[j][1])

                First[n0x, n0x] += 1 - 2 * x + x * x + y * y
                First[n0x, n1x] += 2 * x - 2 * x * x - 2 * y * y
                First[n0x, n1y] += 2 * y
                First[n0x, n2x] += -2 + 2 * x
                First[n0x, n2y] += -2 * y

                First[n0y, n0y] += 1 - 2 * x + x * x + y * y
                First[n0y, n1x] += -2 * y
                First[n0y, n1y] += 2 * x - 2 * x * x - 2 * y * y
                First[n0y, n2x] += 2 * y
                First[n0y, n2y] += -2 + 2 * x

                First[n1x, n1x] += x * x + y * y
                First[n1x, n2x] += -2 * x
                First[n1x, n2y] += 2 * y

                First[n1y, n1y] += x * x + y * y
                First[n1y, n2x] += -2 * y
                First[n1y, n2y] += -2 * x

                First[n2x, n2x] += 1
                First[n2y, n2y] += 1

        freeSize = 2 * nFreeVerts
        constSize = 2 * nConstraints

        G00 = First[0:freeSize, 0:freeSize]
        G01 = First[0:freeSize, freeSize:freeSize + constSize]
        G10 = First[freeSize:freeSize + constSize, 0:freeSize]

        GPrime = G00 + G00.T
        B = G01 + G10.T

        # invert GPrime
        try:
            GPrimeInv = np.linalg.inv(GPrime)
        except np.linalg.LinAlgError:
            GPrimeInv = np.linalg.pinv(GPrime)

        Final = -GPrimeInv @ B
        self.m_mFirstMatrix = Final  # shape (2*nFreeVerts, 2*nConstraints)

    def PrecomputeScalingMatrices(self, nTriangle):
        t = self.m_vTriangles[nTriangle]

        x01 = float(t.vTriCoords[0][0])
        y01 = float(t.vTriCoords[0][1])
        x12 = float(t.vTriCoords[1][0])
        y12 = float(t.vTriCoords[1][1])
        x20 = float(t.vTriCoords[2][0])
        y20 = float(t.vTriCoords[2][1])

        k1 = x12 * y01 + (-1 + x01) * y12
        k2 = -x12 + x01 * x12 - y01 * y12
        k3 = -y01 + x20 * y01 + x01 * y20
        # k4 not used in original Swift
        k5 = -x01 + x01 * x20 - y01 * y20

        a = -1 + x01
        a1 = (-1 + x01) ** 2 + y01 ** 2
        a2 = (x01 ** 2) + (y01 ** 2)
        b = -1 + x20
        b1 = (-1 + x20) ** 2 + y20 ** 2
        c2 = (x12 ** 2) + (y12 ** 2)

        r1 = 1 + 2 * a * x12 + a1 * (x12 ** 2) - 2 * y01 * y12 + a1 * (y12 ** 2)
        r2 = -(b * x01) - b1 * (x01 ** 2) + y01 * (-(b1 * y01) + y20)
        r3 = -(a * x12) - a1 * (x12 ** 2) + y12 * (y01 - a1 * y12)
        r5 = a * x01 + (y01 ** 2)
        r6 = -(b * y01) - x01 * y20
        r7 = 1 + 2 * b * x01 + b1 * (x01 ** 2) + b1 * (y01 ** 2) - 2 * y01 * y20

        mF = np.zeros((4, 4), dtype=np.float64)
        mF[0, 0] = 2 * a1 + 2 * a1 * c2 + 2 * r7
        mF[0, 1] = 0
        mF[0, 2] = 2 * r2 + 2 * r3 - 2 * r5
        mF[0, 3] = 2 * k1 + 2 * r6 + 2 * y01

        mF[1, 0] = 0
        mF[1, 1] = 2 * a1 + 2 * a1 * c2 + 2 * r7
        mF[1, 2] = -2 * k1 + 2 * k3 - 2 * y01
        mF[1, 3] = 2 * r2 + 2 * r3 - 2 * r5

        mF[2, 0] = 2 * r2 + 2 * r3 - 2 * r5
        mF[2, 1] = -2 * k1 + 2 * k3 - 2 * y01
        mF[2, 2] = 2 * a2 + 2 * a2 * b1 + 2 * r1
        mF[2, 3] = 0

        mF[3, 0] = 2 * k1 - 2 * k3 + 2 * y01
        mF[3, 1] = 2 * r2 + 2 * r3 - 2 * r5
        mF[3, 2] = 0
        mF[3, 3] = 2 * a2 + 2 * a2 * b1 + 2 * r1

        # invert and multiply by -1
        try:
            mFInv = np.linalg.inv(mF)
        except np.linalg.LinAlgError:
            mFInv = np.linalg.pinv(mF)
        mFInv = -mFInv

        mC = np.zeros((4, 6), dtype=np.float64)
        mC[0, 0] = 2 * k2
        mC[0, 1] = -2 * k1
        mC[0, 2] = 2 * (-1 - k5)
        mC[0, 3] = 2 * k3
        mC[0, 4] = 2 * a
        mC[0, 5] = -2 * y01

        mC[1, 0] = 2 * k1
        mC[1, 1] = 2 * k2
        mC[1, 2] = -2 * k3
        mC[1, 3] = 2 * (-1 - k5)
        mC[1, 4] = 2 * y01
        mC[1, 5] = 2 * a

        mC[2, 0] = 2 * (-1 - k2)
        mC[2, 1] = 2 * k1
        mC[2, 2] = 2 * k5
        mC[2, 3] = 2 * r6
        mC[2, 4] = -2 * x01
        mC[2, 5] = 2 * y01

        mC[3, 0] = 2 * k1
        mC[3, 1] = 2 * (-1 - k2)
        mC[3, 2] = -2 * k3
        mC[3, 3] = 2 * k5
        mC[3, 4] = -2 * y01
        mC[3, 5] = -2 * x01

        t.mF = mFInv
        t.mC = mC

    def UpdateScaledTriangle(self, nTriangle):
        t = self.m_vTriangles[nTriangle]
        v0 = self.m_vDeformedVerts[t.nVerts[0]].vPosition.astype(np.float64)
        v1 = self.m_vDeformedVerts[t.nVerts[1]].vPosition.astype(np.float64)
        v2 = self.m_vDeformedVerts[t.nVerts[2]].vPosition.astype(np.float64)
        vDeformed = np.array([v0[0], v0[1], v1[0], v1[1], v2[0], v2[1]], dtype=np.float64)

        mCVec = t.mC @ vDeformed
        vSolution = t.mF @ mCVec  # 4-vector: fitted v0(x,y), v1(x,y)

        vFitted0 = np.array([vSolution[0], vSolution[1]], dtype=np.float64)
        vFitted1 = np.array([vSolution[2], vSolution[3]], dtype=np.float64)

        x01 = float(t.vTriCoords[0][0])
        y01 = float(t.vTriCoords[0][1])
        vFitted01 = vFitted1 - vFitted0
        vFitted01Perp = np.array([vFitted01[1], -vFitted01[0]], dtype=np.float64)
        vFitted2 = vFitted0 + x01 * vFitted01 + y01 * vFitted01Perp

        vOrigV0 = self.m_vInitialVerts[t.nVerts[0]].vPosition.astype(np.float64)
        vOrigV1 = self.m_vInitialVerts[t.nVerts[1]].vPosition.astype(np.float64)
        denom = np.linalg.norm(vFitted01)
        scale = (np.linalg.norm(vOrigV1 - vOrigV0) / denom) if denom > 1e-20 else 1.0

        vFitted0 *= scale
        vFitted1 *= scale
        vFitted2 *= scale

        t.vScaled[0] = vFitted0
        t.vScaled[1] = vFitted1
        t.vScaled[2] = vFitted2

    def ApplyFittingStep(self):
        constraints_vec = sorted(list(self.m_vConstraints), key=lambda c: c.nVertex)
        nVerts = len(self.m_vDeformedVerts)
        nConstraints = len(constraints_vec)
        nFreeVerts = nVerts - nConstraints

        vFX = np.zeros(nVerts, dtype=np.float64)
        vFY = np.zeros(nVerts, dtype=np.float64)

        for t in self.m_vTriangles:
            for j in range(3):
                nA = self.m_vVertexMap[t.nVerts[j]]
                nB = self.m_vVertexMap[t.nVerts[(j + 1) % 3]]

                vDeformedA = t.vScaled[j]
                vDeformedB = t.vScaled[(j + 1) % 3]

                vFX[nA] += -2 * vDeformedA[0] + 2 * vDeformedB[0]
                vFX[nB] += 2 * vDeformedA[0] - 2 * vDeformedB[0]

                vFY[nA] += -2 * vDeformedA[1] + 2 * vDeformedB[1]
                vFY[nB] += 2 * vDeformedA[1] - 2 * vDeformedB[1]

        vF0X = vFX[:nFreeVerts].copy()
        vF0Y = vFY[:nFreeVerts].copy()

        vQX = np.array([c.vConstrainedPos[0] for c in constraints_vec], dtype=np.float64)
        vQY = np.array([c.vConstrainedPos[1] for c in constraints_vec], dtype=np.float64)

        rhsX = self.mDX @ vQX
        rhsX += vF0X
        rhsX *= -1.0

        rhsY = self.mDY @ vQY
        rhsY += vF0Y
        rhsY *= -1.0

        # Solve via pre-inverted HXPrime/HYPrime (for simplicity)
        solX = self.mLUDecompX @ rhsX
        solY = self.mLUDecompY @ rhsY

        for i in range(nVerts):
            c = self.Constraint(i, [0.0, 0.0])
            if c in self.m_vConstraints:
                continue
            row = self.m_vVertexMap[i]
            self.m_vDeformedVerts[i].vPosition[0] = float(solX[row])
            self.m_vDeformedVerts[i].vPosition[1] = float(solY[row])

    def ValidateDeformedMesh(self, bRigid):
        nConstraints = len(self.m_vConstraints)
        if nConstraints < 2:
            return

        self.ValidateSetup()

        constraints_vec = sorted(list(self.m_vConstraints), key=lambda c: c.nVertex)
        nVerts = len(self.m_vDeformedVerts)
        nFreeVerts = nVerts - nConstraints

        vQ = np.zeros(2 * nConstraints, dtype=np.float64)
        for i, c in enumerate(constraints_vec):
            vQ[2 * i] = float(c.vConstrainedPos[0])
            vQ[2 * i + 1] = float(c.vConstrainedPos[1])

        # u = Final * q
        vU = self.m_mFirstMatrix @ vQ  # size 2*nFreeVerts

        # write back into free vertices
        for i in range(nVerts):
            c = self.Constraint(i, [0.0, 0.0])
            if c in self.m_vConstraints:
                continue
            row = self.m_vVertexMap[i]
            fX = vU[2 * row]
            fY = vU[2 * row + 1]
            self.m_vDeformedVerts[i].vPosition[:] = [float(fX), float(fY)]

        if bRigid:
            for i in range(len(self.m_vTriangles)):
                self.UpdateScaledTriangle(i)
            self.ApplyFittingStep()

    # API to fit with the app’s calls
    def initialize_from_mesh(self, mesh: TriangleMesh):
        self.InitializeFromMesh(mesh)

    def set_deformed_handle(self, handle: int, pos_xy):
        self.SetDeformedHandle(handle, pos_xy)

    def remove_handle(self, handle: int):
        self.RemoveHandle(handle)

    def force_validation(self):
        # Force precompute if needed for current constraints
        self.ValidateSetup()


# --------------- OpenGL Widget ---------------

class DeformGLWidget(QGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.mesh = TriangleMesh()
        self.deformed_mesh = TriangleMesh()
        self.deformer = RigidMeshDeformer2D()
        self.m_bConstraintsValid = False

        self.m_vSelected = set()
        self.m_nSelected = None

        self.viewport = [0, 0, 600, 600]
        self.translate = np.array([0.0, 0.0], dtype=np.float32)
        self.scale = 1.0

        self.make_square_mesh()
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def make_square_mesh(self):
        self.mesh.clear()
        nRowLen = 5
        yStep = 2.0 / float(nRowLen - 1)
        xStep = 2.0 / float(nRowLen - 1)
        for yi in range(nRowLen):
            y = -1.0 + yi * yStep
            for xi in range(nRowLen):
                x = -1.0 + xi * xStep
                self.mesh.append_vertex([x, y, 0.0])
        for yi in range(nRowLen - 1):
            row1 = yi * nRowLen
            row2 = (yi + 1) * nRowLen
            for xi in range(nRowLen - 1):
                tri1 = [row1 + xi, row2 + xi + 1, row1 + xi + 1]
                tri2 = [row1 + xi, row2 + xi, row2 + xi + 1]
                self.mesh.append_triangle(tri1)
                self.mesh.append_triangle(tri2)
        self.initialize_deformed_mesh()
        self.update()

    def initialize_deformed_mesh(self):
        self.deformed_mesh = TriangleMesh()
        for i in range(self.mesh.get_num_vertices()):
            self.deformed_mesh.append_vertex(self.mesh.get_vertex(i))
        for i in range(self.mesh.get_num_triangles()):
            self.deformed_mesh.append_triangle(self.mesh.get_triangle_indices(i))
        self.deformer.initialize_from_mesh(self.mesh)
        self.invalidate_constraints()

    def invalidate_constraints(self):
        self.m_bConstraintsValid = False

    def validate_constraints(self):
        if self.m_bConstraintsValid:
            return
        for vidx in self.m_vSelected:
            v = self.deformed_mesh.get_vertex(vidx)
            self.deformer.set_deformed_handle(vidx, (v[0], v[1]))
        self.deformer.force_validation()
        self.m_bConstraintsValid = True

    def update_deformed_mesh(self):
        self.validate_constraints()
        self.deformer.UpdateDeformedMesh(self.deformed_mesh, bRigid=True)

    def update_scale(self):
        self.viewport = [0, 0, self.width(), self.height()]
        bounds = self.mesh.get_bounding_box()
        self.translate[0] = (self.viewport[2] / 2.0) - 0.5 * (bounds[0] + bounds[1])
        self.translate[1] = (self.viewport[3] / 2.0) - 0.5 * (bounds[2] + bounds[3])
        width = bounds[1] - bounds[0]
        height = bounds[3] - bounds[2]
        size_obj = max(width, height) if max(width, height) > 0 else 1.0
        size_view = min(self.viewport[2], self.viewport[3]) if min(self.viewport[2], self.viewport[3]) > 0 else 1.0
        self.scale = 0.5 * (size_view / size_obj)

    def world_to_view(self, p):
        return p * self.scale + self.translate

    def view_to_world(self, p):
        return (p - self.translate) / self.scale

    def find_hit_vertex(self, x, y):
        nVerts = self.deformed_mesh.get_num_vertices()
        for i in range(nVerts):
            v = self.deformed_mesh.get_vertex(i)
            view = self.world_to_view(np.array([v[0], v[1]], dtype=np.float32))
            dx = x - view[0]
            dy = y - view[1]
            if math.hypot(dx, dy) < 5:
                return i
        return None

    def initializeGL(self):
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glDisable(GL_DEPTH_TEST)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0, w, 0, h)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def paintGL(self):
        self.update_deformed_mesh()
        glClear(GL_COLOR_BUFFER_BIT)
        self.update_scale()

        glLoadIdentity()
        glTranslatef(float(self.translate[0]), float(self.translate[1]), 0.0)
        glScalef(float(self.scale), float(self.scale), 1.0)

        glLineWidth(2.0)
        glColor3f(0.0, 0.0, 0.0)

        nTris = self.deformed_mesh.get_num_triangles()
        for i in range(nTris):
            v = self.deformed_mesh.get_triangle_vertices(i)
            glBegin(GL_LINE_LOOP)
            glVertex3f(float(v[0, 0]), float(v[0, 1]), float(v[0, 2]))
            glVertex3f(float(v[1, 0]), float(v[1, 1]), float(v[1, 2]))
            glVertex3f(float(v[2, 0]), float(v[2, 1]), float(v[2, 2]))
            glEnd()

        glLoadIdentity()
        glColor3f(1.0, 0.0, 0.0)
        for idx in self.m_vSelected:
            v = self.deformed_mesh.get_vertex(idx)
            view = self.world_to_view(np.array([v[0], v[1]], dtype=np.float32))
            x, y = float(view[0]), float(view[1])
            glBegin(GL_QUADS)
            glVertex2f(x - 5, y - 5)
            glVertex2f(x + 5, y - 5)
            glVertex2f(x + 5, y + 5)
            glVertex2f(x - 5, y + 5)
            glEnd()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            x = event.x()
            y = self.viewport[3] - 1 - event.y()
            self.m_nSelected = self.find_hit_vertex(x, y)
            self.update()
        elif event.button() == QtCore.Qt.RightButton:
            x = event.x()
            y = self.viewport[3] - 1 - event.y()
            hit = self.find_hit_vertex(x, y)
            if hit is not None:
                if hit not in self.m_vSelected:
                    self.m_vSelected.add(hit)
                else:
                    self.m_vSelected.remove(hit)
                    self.deformer.remove_handle(hit)
                    orig = self.mesh.get_vertex(hit)
                    self.deformed_mesh.set_vertex(hit, orig)
                self.invalidate_constraints()
                self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.m_nSelected = None
            self.update()

    def mouseMoveEvent(self, event):
        if self.m_nSelected is not None:
            x = event.x()
            y = self.viewport[3] - 1 - event.y()
            new_pos_view = np.array([x, y], dtype=np.float32)
            new_pos_world = self.view_to_world(new_pos_view)
            self.deformed_mesh.set_vertex(self.m_nSelected, np.array([new_pos_world[0], new_pos_world[1], 0.0], dtype=np.float32))
            self.invalidate_constraints()
            self.update()

    def keyPressEvent(self, event):
        key = event.key()
        if key == QtCore.Qt.Key_F:
            fname, _ = QFileDialog.getOpenFileName(self, "Open OBJ", "", "OBJ Files (*.obj)")
            if fname:
                self.mesh.read_obj(fname)
                self.m_vSelected.clear()
                self.initialize_deformed_mesh()
                self.update()
        else:
            super().keyPressEvent(event)


# --------------- Main Window ---------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deform2D (PyQt5)")
        self.resize(600, 600)
        self.glw = DeformGLWidget(self)
        self.setCentralWidget(self.glw)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()