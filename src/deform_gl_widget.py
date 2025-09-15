import math

import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import *
from PyQt5.QtOpenGL import QGLWidget

from .rigid_mesh_deformer import RigidMeshDeformer
from .triangle_mesh import TriangleMesh

class DeformGLWidget(QGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.mesh = TriangleMesh()
        self.deformed_mesh = TriangleMesh()
        self.deformer = RigidMeshDeformer()
        self.m_bConstraintsValid = False

        self.m_vSelected = set()
        self.m_nSelected = None

        self.viewport = [0, 0, 600, 600]
        self.translate = np.array([0.0, 0.0], dtype=np.float32)
        self.scale = 1.0

        self.load_man_mesh()
        self.setFocusPolicy(Qt.StrongFocus)

    def load_man_mesh(self):
        try:
            self.mesh.read_obj("assets/man.obj")
            self.initialize_deformed_mesh()
            self.update()
        except Exception as e:
            print(f"Error loading man mesh: {e}")
            self.make_square_mesh()

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
        self.deformer.validate_setup()
        self.m_bConstraintsValid = True

    def update_deformed_mesh(self):
        self.validate_constraints()
        self.deformer.update_deformed_mesh(self.deformed_mesh, bRigid=True)

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
        if event.button() == Qt.LeftButton:
            x = event.x()
            y = self.viewport[3] - 1 - event.y()
            self.m_nSelected = self.find_hit_vertex(x, y)
            self.update()
        elif event.button() == Qt.RightButton:
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
        if event.button() == Qt.LeftButton:
            self.m_nSelected = None
            self.update()

    def mouseMoveEvent(self, event):
        if self.m_nSelected is not None:
            x = event.x()
            y = self.viewport[3] - 1 - event.y()
            new_pos_view = np.array([x, y], dtype=np.float32)
            new_pos_world = self.view_to_world(new_pos_view)
            self.deformed_mesh.set_vertex(self.m_nSelected,
                                          np.array([new_pos_world[0], new_pos_world[1], 0.0], dtype=np.float32))
            self.invalidate_constraints()
            self.update()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_F:
            fname, _ = QFileDialog.getOpenFileName(self, "Open OBJ", "", "OBJ Files (*.obj)")
            if fname:
                self.mesh.read_obj(fname)
                self.m_vSelected.clear()
                self.initialize_deformed_mesh()
                self.update()
        else:
            super().keyPressEvent(event)
