"""
Unit tests for Camera3D class
"""
import numpy as np
import pytest
import math


# Import Camera3D from the widget file
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.deform_gl_widget_3d import Camera3D


class TestCamera3DBasics:
    """Test basic Camera3D operations"""
    
    def test_init(self):
        cam = Camera3D()
        assert cam.rotation.shape == (2,)
        assert cam.rotation[0] == 0.0  # pitch
        assert cam.rotation[1] == 0.0  # yaw
        assert cam.distance == 2.0
        assert np.allclose(cam.center, [0.0, 0.0, 0.0])
        assert cam.fov == 45.0
    
    def test_set_rotation(self):
        cam = Camera3D()
        cam.rotation = np.array([30.0, 45.0], dtype=np.float32)
        assert cam.rotation[0] == 30.0
        assert cam.rotation[1] == 45.0
    
    def test_set_distance(self):
        cam = Camera3D()
        cam.distance = 5.0
        assert cam.distance == 5.0
    
    def test_set_center(self):
        cam = Camera3D()
        cam.center = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert np.allclose(cam.center, [1.0, 2.0, 3.0])


class TestViewMatrix:
    """Test view matrix calculations"""
    
    def test_view_matrix_default(self):
        cam = Camera3D()
        eye, center, up = cam.get_view_matrix()
        
        assert eye.shape == (3,)
        assert center.shape == (3,)
        assert up.shape == (3,)
        
        # Default should look from positive Z
        assert eye[2] > 0
        assert np.allclose(center, [0.0, 0.0, 0.0])
        assert np.allclose(up, [0.0, 1.0, 0.0])
    
    def test_view_matrix_pitch_only(self):
        cam = Camera3D()
        cam.rotation = np.array([30.0, 0.0], dtype=np.float32)
        eye, center, up = cam.get_view_matrix()
        
        # With positive pitch, camera should be above the origin
        assert eye[1] > 0
    
    def test_view_matrix_yaw_only(self):
        cam = Camera3D()
        cam.rotation = np.array([0.0, 90.0], dtype=np.float32)
        eye, center, up = cam.get_view_matrix()
        
        # With 90 degree yaw, camera should be on negative X axis
        assert abs(eye[0]) > 1.0  # Should be offset in X
    
    def test_view_matrix_distance(self):
        cam = Camera3D()
        cam.distance = 5.0
        eye, center, up = cam.get_view_matrix()
        
        # Distance from eye to center should be 5.0
        distance = np.linalg.norm(eye - center)
        assert abs(distance - 5.0) < 1e-5
    
    def test_view_matrix_center_offset(self):
        cam = Camera3D()
        cam.center = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        eye, center, up = cam.get_view_matrix()
        
        # Center should be at the specified position
        assert np.allclose(center, [1.0, 2.0, 3.0])
    
    def test_view_matrix_up_vector(self):
        cam = Camera3D()
        eye, center, up = cam.get_view_matrix()
        
        # Up vector should always be [0, 1, 0]
        assert np.allclose(up, [0.0, 1.0, 0.0])


class TestCameraRotation:
    """Test camera rotation calculations"""
    
    def test_rotation_0_0(self):
        """Pitch=0, Yaw=0 should look along -Z axis"""
        cam = Camera3D()
        cam.rotation = np.array([0.0, 0.0], dtype=np.float32)
        cam.distance = 1.0
        eye, center, up = cam.get_view_matrix()
        
        # Camera should be at (0, 0, 1) looking at (0, 0, 0)
        assert abs(eye[0]) < 1e-5
        assert abs(eye[1]) < 1e-5
        assert abs(eye[2] - 1.0) < 1e-5
    
    def test_rotation_90_pitch(self):
        """Pitch=90 would be looking straight down (but clamped to 89)"""
        cam = Camera3D()
        cam.rotation = np.array([89.0, 0.0], dtype=np.float32)
        cam.distance = 1.0
        eye, center, up = cam.get_view_matrix()
        
        # Camera should be almost directly above
        assert eye[1] > 0.9  # Y should be close to 1
    
    def test_rotation_180_yaw(self):
        """Yaw=180 should look along +Z axis"""
        cam = Camera3D()
        cam.rotation = np.array([0.0, 180.0], dtype=np.float32)
        cam.distance = 1.0
        eye, center, up = cam.get_view_matrix()
        
        # Camera should be at (0, 0, -1) looking at (0, 0, 0)
        assert abs(eye[0]) < 1e-5
        assert abs(eye[1]) < 1e-5
        assert abs(eye[2] + 1.0) < 1e-5
    
    def test_rotation_45_45(self):
        """Test isometric-like view"""
        cam = Camera3D()
        cam.rotation = np.array([45.0, 45.0], dtype=np.float32)
        cam.distance = math.sqrt(3)  # Distance for unit cube
        eye, center, up = cam.get_view_matrix()
        
        # All coordinates should be non-zero for isometric view
        assert abs(eye[0]) > 0.1
        assert abs(eye[1]) > 0.1
        assert abs(eye[2]) > 0.1


class TestCameraDistance:
    """Test camera distance calculations"""
    
    def test_distance_changes_eye_position(self):
        cam = Camera3D()
        cam.rotation = np.array([0.0, 0.0], dtype=np.float32)
        
        cam.distance = 1.0
        eye1, _, _ = cam.get_view_matrix()
        
        cam.distance = 2.0
        eye2, _, _ = cam.get_view_matrix()
        
        # Eye position should scale with distance
        assert np.linalg.norm(eye2) > np.linalg.norm(eye1)
    
    def test_distance_preserves_direction(self):
        cam = Camera3D()
        cam.rotation = np.array([30.0, 45.0], dtype=np.float32)
        
        cam.distance = 1.0
        eye1, center1, _ = cam.get_view_matrix()
        dir1 = (eye1 - center1) / np.linalg.norm(eye1 - center1)
        
        cam.distance = 5.0
        eye2, center2, _ = cam.get_view_matrix()
        dir2 = (eye2 - center2) / np.linalg.norm(eye2 - center2)
        
        # Direction should be the same
        assert np.allclose(dir1, dir2, atol=1e-5)


class TestCameraEdgeCases:
    """Test edge cases and extreme values"""
    
    def test_zero_distance(self):
        """Zero distance should not crash"""
        cam = Camera3D()
        cam.distance = 0.0
        eye, center, up = cam.get_view_matrix()
        
        # Should return valid values
        assert not np.any(np.isnan(eye))
        assert not np.any(np.isnan(center))
        assert not np.any(np.isnan(up))
    
    def test_very_large_distance(self):
        """Very large distance should work"""
        cam = Camera3D()
        cam.distance = 1000.0
        eye, center, up = cam.get_view_matrix()
        
        distance = np.linalg.norm(eye - center)
        assert abs(distance - 1000.0) < 1e-3
    
    def test_extreme_pitch_positive(self):
        """Test maximum positive pitch"""
        cam = Camera3D()
        cam.rotation = np.array([89.0, 0.0], dtype=np.float32)
        eye, center, up = cam.get_view_matrix()
        
        assert not np.any(np.isnan(eye))
    
    def test_extreme_pitch_negative(self):
        """Test maximum negative pitch"""
        cam = Camera3D()
        cam.rotation = np.array([-89.0, 0.0], dtype=np.float32)
        eye, center, up = cam.get_view_matrix()
        
        assert not np.any(np.isnan(eye))
    
    def test_full_yaw_rotation(self):
        """Test yaw values around full circle"""
        cam = Camera3D()
        cam.distance = 1.0
        
        for yaw in [0, 90, 180, 270, 360]:
            cam.rotation = np.array([0.0, float(yaw)], dtype=np.float32)
            eye, center, up = cam.get_view_matrix()
            
            # Should always be at distance 1.0 from center
            distance = np.linalg.norm(eye - center)
            assert abs(distance - 1.0) < 1e-5
    
    def test_combined_extreme_rotation(self):
        """Test combined extreme rotations"""
        cam = Camera3D()
        cam.rotation = np.array([89.0, 359.0], dtype=np.float32)
        eye, center, up = cam.get_view_matrix()
        
        assert not np.any(np.isnan(eye))
        assert not np.any(np.isnan(center))
