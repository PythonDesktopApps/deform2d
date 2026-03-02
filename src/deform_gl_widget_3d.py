import math
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import *
from PyQt5.QtOpenGL import QGLWidget

from .rigid_mesh_deformer import RigidMeshDeformer
from .triangle_mesh import TriangleMesh


class Camera3D:
    """3D camera with trackball-style rotation"""
    def __init__(self):
        self.rotation = np.array([0.0, 0.0], dtype=np.float32)  # pitch, yaw
        self.distance = 2.0
        self.center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.fov = 45.0
        
    def get_view_matrix(self):
        """Get the view transformation matrix"""
        # This returns parameters for gluLookAt
        # Position camera based on rotation and distance
        pitch_rad = np.radians(self.rotation[0])
        yaw_rad = np.radians(self.rotation[1])
        
        cam_x = self.distance * np.cos(pitch_rad) * np.sin(yaw_rad)
        cam_y = self.distance * np.sin(pitch_rad)
        cam_z = self.distance * np.cos(pitch_rad) * np.cos(yaw_rad)
        
        eye = self.center + np.array([cam_x, cam_y, cam_z], dtype=np.float32)
        return eye, self.center, np.array([0.0, 1.0, 0.0], dtype=np.float32)


class DeformGLWidget3D(QGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.mesh = TriangleMesh()
        self.deformer = RigidMeshDeformer()
        self.m_bConstraintsValid = False

        self.m_vSelected = set()
        self.m_nSelected = None
        
        # 3D camera
        self.camera = Camera3D()
        self.is_3d_mesh = False
        
        # Rotation mode: 'object' or 'camera'
        self.rotation_mode = 'object'  # Default to rotating object
        self.object_rotation = np.array([0.0, 180.0], dtype=np.float32)  # pitch, yaw
        
        # Mouse interaction
        self.last_mouse_pos = None
        self.is_rotating = False
        self.drag_plane_point = None  # Point on drag plane
        self.drag_plane_normal = None  # Normal of drag plane
        
        # Viewport
        self.viewport = [0, 0, 600, 600]
        
        # Projection matrices (cached)
        self.projection_matrix = None
        self.modelview_matrix = None

        self.load_default_mesh()
        self.setFocusPolicy(Qt.StrongFocus)

    def load_default_mesh(self):
        """Load the default mesh (try armadillo first, fall back to man.obj)"""
        try:
            self.mesh.read_off("assets/armadillo_250.off")
            self.is_3d_mesh = True
            self.initialize_deformed_mesh()
            self.fit_camera_to_mesh()
            self.update()
        except Exception as e:
            print(f"Error loading armadillo mesh: {e}")
            try:
                self.mesh.read_obj("assets/man.obj")
                self.is_3d_mesh = self.check_if_3d()
                self.initialize_deformed_mesh()
                self.fit_camera_to_mesh()
                self.update()
            except Exception as e2:
                print(f"Error loading man mesh: {e2}")
                self.make_square_mesh()

    def check_if_3d(self):
        """Check if the mesh is actually 3D (has non-zero z values)"""
        if self.mesh.get_num_vertices() == 0:
            return False
        z_vals = self.mesh.vertices[:, 2]
        return np.any(np.abs(z_vals) > 1e-6)

    def reset_view(self):
        """Reset only the camera/object rotation to default angles (doesn't change zoom/position)"""
        if not self.is_3d_mesh:
            # For 2D meshes, look straight down
            self.camera.rotation = np.array([0.0, 0.0], dtype=np.float32)
            self.object_rotation = np.array([0.0, 0.0], dtype=np.float32)
        else:
            # For 3D meshes, set to nice 3/4 view
            self.object_rotation = np.array([0.0, 180.0], dtype=np.float32)
            self.camera.rotation = np.array([0.0, 0.0], dtype=np.float32)
        self.update()
    
    def fit_camera_to_mesh(self):
        """Adjust camera to fit the mesh in view (resets zoom, position, and rotation)"""
        if self.mesh.get_num_vertices() == 0:
            return
        
        bounds = self.mesh.get_bounding_box()
        center = np.array([
            (bounds[0] + bounds[1]) / 2.0,
            (bounds[2] + bounds[3]) / 2.0,
            (bounds[4] + bounds[5]) / 2.0
        ], dtype=np.float32)
        
        size = max(
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4]
        )
        
        self.camera.center = center
        self.camera.distance = size * 2.0 if size > 0 else 2.0
        
        # Also reset rotation
        self.reset_view()

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
        self.is_3d_mesh = False
        self.initialize_deformed_mesh()
        self.fit_camera_to_mesh()
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
            self.deformer.set_deformed_handle(vidx, (v[0], v[1], v[2]))
        self.deformer.validate_setup()
        self.m_bConstraintsValid = True

    def update_deformed_mesh(self):
        self.validate_constraints()
        self.deformer.update_deformed_mesh(self.deformed_mesh, bRigid=True)

    def project_point(self, point_3d):
        """Project a 3D point to screen coordinates"""
        if self.modelview_matrix is None or self.projection_matrix is None:
            return None
        
        result = gluProject(
            point_3d[0], point_3d[1], point_3d[2],
            self.modelview_matrix,
            self.projection_matrix,
            self.viewport
        )
        
        if result is None:
            return None
        
        return np.array([result[0], result[1], result[2]], dtype=np.float32)

    def unproject_point(self, screen_x, screen_y, depth):
        """Unproject screen coordinates to 3D world space"""
        if self.modelview_matrix is None or self.projection_matrix is None:
            return None
        
        result = gluUnProject(
            screen_x, screen_y, depth,
            self.modelview_matrix,
            self.projection_matrix,
            self.viewport
        )
        
        if result is None:
            return None
        
        return np.array([result[0], result[1], result[2]], dtype=np.float32)

    def get_ray_from_screen(self, screen_x, screen_y):
        """Get a ray from camera through screen point"""
        # Unproject near and far points
        near_point = self.unproject_point(screen_x, screen_y, 0.0)
        far_point = self.unproject_point(screen_x, screen_y, 1.0)
        
        if near_point is None or far_point is None:
            return None, None
        
        ray_origin = near_point
        ray_dir = far_point - near_point
        ray_dir = ray_dir / np.linalg.norm(ray_dir)
        
        return ray_origin, ray_dir

    def find_hit_vertex(self, screen_x, screen_y):
        """Find the vertex closest to the ray from screen coordinates"""
        ray_origin, ray_dir = self.get_ray_from_screen(screen_x, screen_y)
        if ray_origin is None or ray_dir is None:
            return None
        
        nVerts = self.deformed_mesh.get_num_vertices()
        best_vertex = None
        best_distance = float('inf')
        threshold = 0.05  # World space threshold for picking
        
        for i in range(nVerts):
            v = self.deformed_mesh.get_vertex(i)
            
            # Calculate distance from point to ray
            to_point = v - ray_origin
            projection_length = np.dot(to_point, ray_dir)
            
            if projection_length < 0:
                continue  # Behind camera
            
            closest_point_on_ray = ray_origin + projection_length * ray_dir
            distance = np.linalg.norm(v - closest_point_on_ray)
            
            if distance < threshold and projection_length < best_distance:
                best_distance = projection_length
                best_vertex = i
        
        return best_vertex

    def setup_drag_plane(self, vertex_idx):
        """Setup a plane for dragging the selected vertex"""
        v = self.deformed_mesh.get_vertex(vertex_idx)
        
        # Get camera direction
        eye, center, up = self.camera.get_view_matrix()
        view_dir = center - eye
        view_dir = view_dir / np.linalg.norm(view_dir)
        
        # Drag plane is perpendicular to view direction, passing through vertex
        self.drag_plane_point = v.copy()
        self.drag_plane_normal = view_dir.copy()

    def intersect_ray_plane(self, ray_origin, ray_dir):
        """Find intersection of ray with drag plane"""
        if self.drag_plane_point is None or self.drag_plane_normal is None:
            return None
        
        denom = np.dot(ray_dir, self.drag_plane_normal)
        
        if abs(denom) < 1e-6:
            return None  # Ray parallel to plane
        
        t = np.dot(self.drag_plane_point - ray_origin, self.drag_plane_normal) / denom
        
        if t < 0:
            return None  # Intersection behind ray origin
        
        return ray_origin + t * ray_dir

    def initializeGL(self):
        glClearColor(0.9, 0.9, 0.95, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_MULTISAMPLE)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        
        # Lighting setup
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [1.0, 1.0, 1.0, 0.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.7, 0.7, 0.7, 1.0])
        
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    def resizeGL(self, w, h):
        self.viewport = [0, 0, w, h]
        glViewport(0, 0, w, h)

    def paintGL(self):
        self.update_deformed_mesh()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Setup projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = self.viewport[2] / max(self.viewport[3], 1)
        gluPerspective(self.camera.fov, aspect, 0.01, 100.0)
        self.projection_matrix = glGetDoublev(GL_PROJECTION_MATRIX)
        
        # Setup modelview
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        eye, center, up = self.camera.get_view_matrix()
        gluLookAt(eye[0], eye[1], eye[2],
                  center[0], center[1], center[2],
                  up[0], up[1], up[2])
        
        # Apply object rotation if in object rotation mode
        if self.rotation_mode == 'object':
            # Translate to center, rotate, translate back
            glTranslatef(self.camera.center[0], self.camera.center[1], self.camera.center[2])
            glRotatef(-self.object_rotation[0], 1.0, 0.0, 0.0)  # Pitch (around X axis)
            glRotatef(-self.object_rotation[1], 0.0, 1.0, 0.0)  # Yaw (around Y axis)
            glTranslatef(-self.camera.center[0], -self.camera.center[1], -self.camera.center[2])
        
        self.modelview_matrix = glGetDoublev(GL_MODELVIEW_MATRIX)
        
        # Draw mesh
        self.draw_mesh()
        
        # Draw selected vertices
        self.draw_selected_vertices()

    def draw_mesh(self):
        """Draw the deformed mesh"""
        glDisable(GL_LIGHTING)
        glLineWidth(1.5)
        glColor3f(0.2, 0.2, 0.2)
        
        nTris = self.deformed_mesh.get_num_triangles()
        for i in range(nTris):
            v = self.deformed_mesh.get_triangle_vertices(i)
            glBegin(GL_LINE_LOOP)
            glVertex3f(float(v[0, 0]), float(v[0, 1]), float(v[0, 2]))
            glVertex3f(float(v[1, 0]), float(v[1, 1]), float(v[1, 2]))
            glVertex3f(float(v[2, 0]), float(v[2, 1]), float(v[2, 2]))
            glEnd()
        
        # Optionally draw filled triangles
        glEnable(GL_LIGHTING)
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(1.0, 1.0)
        glColor3f(0.8, 0.85, 0.9)
        
        glBegin(GL_TRIANGLES)
        for i in range(nTris):
            v = self.deformed_mesh.get_triangle_vertices(i)
            # Calculate normal
            e1 = v[1] - v[0]
            e2 = v[2] - v[0]
            normal = np.cross(e1, e2)
            norm = np.linalg.norm(normal)
            if norm > 1e-6:
                normal = normal / norm
            glNormal3f(float(normal[0]), float(normal[1]), float(normal[2]))
            
            glVertex3f(float(v[0, 0]), float(v[0, 1]), float(v[0, 2]))
            glVertex3f(float(v[1, 0]), float(v[1, 1]), float(v[1, 2]))
            glVertex3f(float(v[2, 0]), float(v[2, 1]), float(v[2, 2]))
        glEnd()
        
        glDisable(GL_POLYGON_OFFSET_FILL)
        glDisable(GL_LIGHTING)

    def draw_selected_vertices(self):
        """Draw markers for selected (pinned) vertices"""
        glDisable(GL_DEPTH_TEST)
        glPointSize(10.0)
        glColor3f(1.0, 0.0, 0.0)
        
        glBegin(GL_POINTS)
        for idx in self.m_vSelected:
            v = self.deformed_mesh.get_vertex(idx)
            glVertex3f(float(v[0]), float(v[1]), float(v[2]))
        glEnd()
        
        glEnable(GL_DEPTH_TEST)

    def mousePressEvent(self, event):
        x = event.x()
        y = self.viewport[3] - 1 - event.y()
        
        if event.button() == Qt.LeftButton:
            hit = self.find_hit_vertex(x, y)
            if hit is not None:
                self.m_nSelected = hit
                self.setup_drag_plane(hit)
            else:
                # Start rotation
                self.is_rotating = True
                self.last_mouse_pos = np.array([event.x(), event.y()], dtype=np.float32)
            self.update()
            
        elif event.button() == Qt.RightButton:
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
            self.is_rotating = False
            self.drag_plane_point = None
            self.drag_plane_normal = None
            self.update()

    def mouseMoveEvent(self, event):
        if self.is_rotating and self.last_mouse_pos is not None:
            # Rotate camera or object based on mode
            current_pos = np.array([event.x(), event.y()], dtype=np.float32)
            delta = current_pos - self.last_mouse_pos
            
            if self.rotation_mode == 'camera':
                # Camera rotation: drag left -> camera moves left -> object appears to rotate right
                self.camera.rotation[1] += delta[0] * 0.5  # Yaw
                self.camera.rotation[0] += delta[1] * 0.5  # Pitch
                # Clamp pitch
                self.camera.rotation[0] = np.clip(self.camera.rotation[0], -89.0, 89.0)
            else:  # object rotation
                # Object rotation: drag left -> object rotates left (intuitive)
                # Negate the delta to make rotation follow mouse direction
                self.object_rotation[1] -= delta[0] * 0.5  # Yaw (inverted)
                self.object_rotation[0] -= delta[1] * 0.5  # Pitch (inverted)
                # Clamp pitch
                self.object_rotation[0] = np.clip(self.object_rotation[0], -89.0, 89.0)
            
            self.last_mouse_pos = current_pos
            self.update()
            
        elif self.m_nSelected is not None:
            # Drag vertex
            x = event.x()
            y = self.viewport[3] - 1 - event.y()
            
            ray_origin, ray_dir = self.get_ray_from_screen(x, y)
            if ray_origin is not None and ray_dir is not None:
                new_pos = self.intersect_ray_plane(ray_origin, ray_dir)
                if new_pos is not None:
                    self.deformed_mesh.set_vertex(self.m_nSelected, new_pos)
                    self.invalidate_constraints()
                    self.update()

    def wheelEvent(self, event):
        # Zoom with mouse wheel
        delta = event.angleDelta().y()
        zoom_factor = 1.1 if delta > 0 else 0.9
        self.camera.distance *= zoom_factor
        self.camera.distance = max(0.1, min(self.camera.distance, 100.0))
        self.update()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_F:
            fname, _ = QFileDialog.getOpenFileName(
                self, "Open Mesh", "",
                "Mesh Files (*.obj *.off);;OBJ Files (*.obj);;OFF Files (*.off);;All Files (*.*)"
            )
            if fname:
                try:
                    if fname.lower().endswith('.off'):
                        self.mesh.read_off(fname)
                    else:
                        self.mesh.read_obj(fname)

                    self.is_3d_mesh = self.check_if_3d()
                    self.m_vSelected.clear()
                    self.initialize_deformed_mesh()
                    self.fit_camera_to_mesh()
                    self.update()
                except Exception as e:
                    print(f"Error loading mesh: {e}")

        elif key == Qt.Key_R:
            # Reset camera
            self.fit_camera_to_mesh()
            self.update()

        elif key == Qt.Key_C:
            # Clear constraints
            self.m_vSelected.clear()
            self.deformer = RigidMeshDeformer()
            self.initialize_deformed_mesh()
            self.update()

        else:
            super().keyPressEvent(event)
