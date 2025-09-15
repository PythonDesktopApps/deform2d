import sys
from PyQt5.QtWidgets import QMainWindow, QApplication
from src.deform_gl_widget import DeformGLWidget


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
