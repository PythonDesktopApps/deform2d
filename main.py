import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QStatusBar, QWidget, QHBoxLayout
from src.deform_gl_widget_3d import DeformGLWidget3D
from src.control_panel import ControlPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ARAP Mesh Deformation (2D/3D) - PyQt5 + PyOpenGL")
        self.resize(1000, 600)
        
        # Create central widget with horizontal layout
        central_widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create control panel
        self.control_panel = ControlPanel()
        layout.addWidget(self.control_panel)
        
        # Create GL widget
        self.glw = DeformGLWidget3D(self)
        layout.addWidget(self.glw, stretch=1)
        
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        
        # Connect control panel signals to GL widget
        self.control_panel.rotation_mode_changed.connect(self.on_rotation_mode_changed)
        self.control_panel.reset_view_requested.connect(self.glw.reset_view)
        self.control_panel.clear_constraints_requested.connect(self.clear_constraints)
        self.control_panel.load_mesh_requested.connect(self.load_mesh)
        
        # Add status bar with instructions
        status = QStatusBar()
        status.showMessage("Right-click: Pin/Unpin vertex | Left-click + drag: Deform or Rotate | Wheel: Zoom")
        self.setStatusBar(status)
    
    def on_rotation_mode_changed(self, mode):
        """Handle rotation mode change from control panel"""
        self.glw.rotation_mode = mode
        self.glw.update()
    
    def clear_constraints(self):
        """Clear all constraints, reset mesh, and reset camera view"""
        self.glw.m_vSelected.clear()
        from src.rigid_mesh_deformer import RigidMeshDeformer
        self.glw.deformer = RigidMeshDeformer()
        self.glw.initialize_deformed_mesh()
        self.glw.fit_camera_to_mesh()  # Also reset camera/view
        self.glw.update()
    
    def load_mesh(self):
        """Trigger file dialog to load mesh"""
        from PyQt5.QtWidgets import QFileDialog
        fname, _ = QFileDialog.getOpenFileName(
            self, "Open Mesh", "", 
            "Mesh Files (*.obj *.off);;OBJ Files (*.obj);;OFF Files (*.off);;All Files (*.*)"
        )
        if fname:
            try:
                if fname.lower().endswith('.off'):
                    self.glw.mesh.read_off(fname)
                else:
                    self.glw.mesh.read_obj(fname)
                
                self.glw.is_3d_mesh = self.glw.check_if_3d()
                self.glw.m_vSelected.clear()
                self.glw.initialize_deformed_mesh()
                self.glw.fit_camera_to_mesh()
                self.glw.update()
            except Exception as e:
                print(f"Error loading mesh: {e}")
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        from PyQt5.QtCore import Qt
        key = event.key()
        
        if key == Qt.Key_F:
            self.load_mesh()
        elif key == Qt.Key_C:
            self.clear_constraints()
        elif key == Qt.Key_R:
            self.glw.reset_view()
        else:
            super().keyPressEvent(event)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
