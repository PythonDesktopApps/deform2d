from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QRadioButton, 
                             QPushButton, QLabel, QButtonGroup)
from PyQt5.QtCore import pyqtSignal


class ControlPanel(QWidget):
    """Control panel for mesh deformation settings"""
    
    rotation_mode_changed = pyqtSignal(str)  # Emits 'camera' or 'object'
    reset_view_requested = pyqtSignal()
    clear_constraints_requested = pyqtSignal()
    load_mesh_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("<b>Deformation Controls</b>")
        layout.addWidget(title)
        
        # Rotation mode group
        rotation_group = QGroupBox("Rotation Mode")
        rotation_layout = QVBoxLayout()
        
        self.object_rotation_radio = QRadioButton("Rotate Object")
        self.camera_rotation_radio = QRadioButton("Rotate Camera")
        
        # Add tooltips
        self.object_rotation_radio.setToolTip(
            "Rotate the object in the direction you drag.\n"
            "The object spins like you're turning it with your mouse."
        )
        self.camera_rotation_radio.setToolTip(
            "Move the camera around the object.\n"
            "Like orbiting around a stationary sculpture."
        )
        
        # Set object rotation as default
        self.object_rotation_radio.setChecked(True)
        
        # Button group to make them mutually exclusive
        self.rotation_button_group = QButtonGroup()
        self.rotation_button_group.addButton(self.object_rotation_radio)
        self.rotation_button_group.addButton(self.camera_rotation_radio)
        
        rotation_layout.addWidget(self.object_rotation_radio)
        rotation_layout.addWidget(self.camera_rotation_radio)
        rotation_group.setLayout(rotation_layout)
        layout.addWidget(rotation_group)
        
        # Connect signals
        self.object_rotation_radio.toggled.connect(self.on_rotation_mode_changed)
        self.camera_rotation_radio.toggled.connect(self.on_rotation_mode_changed)
        
        # Action buttons
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout()
        
        self.load_mesh_btn = QPushButton("Load Mesh (F)")
        self.reset_view_btn = QPushButton("Reset View (R)")
        self.clear_constraints_btn = QPushButton("Clear All (C)")
        
        # Add tooltips to clarify what each button does
        self.load_mesh_btn.setToolTip("Open a mesh file (.obj or .off)")
        self.reset_view_btn.setToolTip(
            "Reset camera/rotation to default angle.\n"
            "Keeps pins and deformation intact."
        )
        self.clear_constraints_btn.setToolTip(
            "Reset EVERYTHING:\n"
            "• Remove all pins\n"
            "• Reset mesh to original shape\n"
            "• Reset camera view to default"
        )
        
        actions_layout.addWidget(self.load_mesh_btn)
        actions_layout.addWidget(self.reset_view_btn)
        actions_layout.addWidget(self.clear_constraints_btn)
        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)
        
        # Connect action buttons
        self.load_mesh_btn.clicked.connect(self.load_mesh_requested.emit)
        self.reset_view_btn.clicked.connect(self.reset_view_requested.emit)
        self.clear_constraints_btn.clicked.connect(self.clear_constraints_requested.emit)
        
        # Instructions
        instructions_group = QGroupBox("Instructions")
        instructions_layout = QVBoxLayout()
        
        instructions_text = QLabel(
            "<small>"
            "<b>Pin Vertices:</b><br/>"
            "Right-click vertices<br/>"
            "(Need at least 2)<br/><br/>"
            "<b>Deform:</b><br/>"
            "Left-click + drag vertex<br/><br/>"
            "<b>Rotate:</b><br/>"
            "Left-click + drag empty space<br/><br/>"
            "<b>Zoom:</b><br/>"
            "Mouse wheel"
            "</small>"
        )
        instructions_text.setWordWrap(True)
        instructions_layout.addWidget(instructions_text)
        instructions_group.setLayout(instructions_layout)
        layout.addWidget(instructions_group)
        
        # Add stretch to push everything to the top
        layout.addStretch()
        
        self.setLayout(layout)
        self.setMaximumWidth(200)
        self.setMinimumWidth(180)
    
    def on_rotation_mode_changed(self):
        """Handle rotation mode radio button change"""
        if self.object_rotation_radio.isChecked():
            self.rotation_mode_changed.emit('object')
        else:
            self.rotation_mode_changed.emit('camera')
    
    def set_rotation_mode(self, mode):
        """Set the rotation mode programmatically"""
        if mode == 'object':
            self.object_rotation_radio.setChecked(True)
        else:
            self.camera_rotation_radio.setChecked(True)
