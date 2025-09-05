import numpy as np
from triangle_mesh import TriangleMesh
from elements import Constraint, Vertex, Triangle
from scipy.linalg import lu_factor, lu_solve

class RigidMeshDeformer:

    def __init__(self):
        self.m_vConstraints = set()
        self.m_vInitialVerts = []
        self.m_vDeformedVerts = []
        self.m_vTriangles = []
        self.m_bSetupValid = False

        self.m_vVertexMap = []
        self.m_mFirstMatrix = None
        self.mHXPrime = None
        self.mHYPrime = None
        self.mDX = None
        self.mDY = None
        self.mLUFactX = None
        self.mLUFactY = None

    def invalidate_setup(self):
        self.m_bSetupValid = False

    def initialize_from_mesh(self, mesh: TriangleMesh):
        self.m_vConstraints.clear()
        self.m_vInitialVerts = []
        self.m_vDeformedVerts = []
        self.m_vTriangles = []

        # copy vertices
        nVerts = mesh.get_num_vertices()
        for i in range(nVerts):
            v3 = mesh.get_vertex(i)
            v = Vertex([v3[0], v3[1]])
            self.m_vInitialVerts.append(Vertex(v.vPosition))
            self.m_vDeformedVerts.append(Vertex(v.vPosition))

        # copy triangles
        nTris = mesh.get_num_triangles()
        for i in range(nTris):
            t = Triangle()
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

        self.invalidate_setup()

    def set_deformed_handle(self, handle: int, pos_xy):
        c = Constraint(handle, pos_xy)
        self.update_constraint(c)

    def remove_handle(self, handle: int):
        c = Constraint(handle, np.array([0.0, 0.0], dtype=np.float32))
        if c in self.m_vConstraints:
            self.m_vConstraints.remove(c)
        # restore to initial
        self.m_vDeformedVerts[handle].vPosition = self.m_vInitialVerts[handle].vPosition.copy()
        self.invalidate_setup()

    def validate_setup(self):
        if self.m_bSetupValid or len(self.m_vConstraints) < 2:
            return

        self.precompute_orientation_matrix()
        for i in range(len(self.m_vTriangles)):
            self.precompute_scaling_matrices(i)
        self.precompute_fitting_matrices()

        self.m_bSetupValid = True

    def untransform_point(self, vTransform):
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

    def update_deformed_mesh(self, mesh: TriangleMesh, bRigid=True):
        self.validate_deformed_mesh(bRigid)
        vVerts = self.m_vDeformedVerts if len(self.m_vConstraints) > 1 else self.m_vInitialVerts
        nVerts = mesh.get_num_vertices()
        arr = mesh.vertices.copy()
        for i in range(nVerts):
            p = vVerts[i].vPosition
            arr[i, 0] = p[0]
            arr[i, 1] = p[1]
            arr[i, 2] = 0.0
        mesh.vertices = arr

    def update_constraint(self, cons):
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
            self.invalidate_setup()

    def extract_submatrix(self, M, rowOffset, colOffset, rows, cols):
        return M[rowOffset:rowOffset + rows, colOffset:colOffset + cols].copy()


    def precompute_fitting_matrices(self, symmetric=False):
        constraints_vec = sorted(list(self.m_vConstraints), key=lambda c: c.nVertex)

        nVerts = len(self.m_vDeformedVerts)
        nConstraints = len(constraints_vec)
        nFreeVerts = nVerts - nConstraints

        # vertex map: free first, then constraints
        self.m_vVertexMap = [-1] * nVerts
        row = 0
        for i in range(nVerts):
            c = Constraint(i, [0.0, 0.0])
            if c in self.m_vConstraints:
                continue
            self.m_vVertexMap[i] = row
            row += 1
        assert row == nFreeVerts
        for i in range(nConstraints):
            self.m_vVertexMap[constraints_vec[i].nVertex] = row
            row += 1
        assert row == nVerts

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

        HX00 = HX[0:nFreeVerts, 0:nFreeVerts]
        HY00 = HY[0:nFreeVerts, 0:nFreeVerts]

        HX01 = HX[0:nFreeVerts, nFreeVerts:nFreeVerts + nConstraints]
        HY01 = HY[0:nFreeVerts, nFreeVerts:nFreeVerts + nConstraints]

        if symmetric:
            HX10 = HX[nFreeVerts:nFreeVerts + nConstraints, 0:nFreeVerts]
            HY10 = HY[nFreeVerts:nFreeVerts + nConstraints, 0:nFreeVerts]
            self.mHXPrime = HX00 + HX00.T
            self.mHYPrime = HY00 + HY00.T
            self.mDX = HX01 + HX10.T
            self.mDY = HY01 + HY10.T
        else:
            self.mHXPrime = HX00.copy()
            self.mHYPrime = HY00.copy()
            self.mDX = HX01.copy()
            self.mDY = HY01.copy()

        # LU factorization with SciPy
        # store (lu, piv) tuples for later lu_solve calls
        self.mLUFactX = lu_factor(self.mHXPrime, check_finite=False)
        self.mLUFactY = lu_factor(self.mHYPrime, check_finite=False)

    def precompute_orientation_matrix(self):
        constraints_vec = sorted(list(self.m_vConstraints), key=lambda c: c.nVertex)
        nVerts = len(self.m_vDeformedVerts)
        nConstraints = len(constraints_vec)
        nFreeVerts = nVerts - nConstraints

        # vertex map: free first, then constraints
        self.m_vVertexMap = [-1] * nVerts
        row = 0
        for i in range(nVerts):
            c = Constraint(i, [0.0, 0.0])
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

    def precompute_scaling_matrices(self, nTriangle):
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

    def update_scaled_triangle(self, nTriangle):
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

    def apply_fitting_step(self):
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

        rhsX = -(self.mDX @ vQX + vF0X)
        rhsY = -(self.mDY @ vQY + vF0Y)

        # Solve using SciPy LU factors
        solX = lu_solve(self.mLUFactX, rhsX, check_finite=False)
        solY = lu_solve(self.mLUFactY, rhsY, check_finite=False)

        for i in range(nVerts):
            c = Constraint(i, [0.0, 0.0])
            if c in self.m_vConstraints:
                continue
            row = self.m_vVertexMap[i]
            self.m_vDeformedVerts[i].vPosition[0] = float(solX[row])
            self.m_vDeformedVerts[i].vPosition[1] = float(solY[row])

    def validate_deformed_mesh(self, bRigid):
        nConstraints = len(self.m_vConstraints)
        if nConstraints < 2:
            return

        self.validate_setup()

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
            c = Constraint(i, [0.0, 0.0])
            if c in self.m_vConstraints:
                continue
            row = self.m_vVertexMap[i]
            fX = vU[2 * row]
            fY = vU[2 * row + 1]
            self.m_vDeformedVerts[i].vPosition[:] = [float(fX), float(fY)]

        if bRigid:
            for i in range(len(self.m_vTriangles)):
                self.update_scaled_triangle(i)
            self.apply_fitting_step()

