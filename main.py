import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QStatusBar
from src.deform_gl_widget_3d import DeformGLWidget3D


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ARAP Mesh Deformation (2D/3D) - PyQt5 + PyOpenGL")
        self.resize(800, 600)
        self.glw = DeformGLWidget3D(self)
        self.setCentralWidget(self.glw)
        
        # Add status bar with instructions
        status = QStatusBar()
        status.showMessage("Right-click: Pin/Unpin vertex | Left-click + drag: Deform mesh or rotate camera | Mouse wheel: Zoom | F: Load mesh | R: Reset camera | C: Clear pins")
        self.setStatusBar(status)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
