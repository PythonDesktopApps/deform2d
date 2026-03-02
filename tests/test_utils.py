"""
Unit tests for utility functions (math operations)
"""
import numpy as np
import pytest
from src.utils import vec2, length2, normalize2, perp, barycentric_coords


class TestVec2:
    """Test vec2 function"""
    
    def test_vec2_from_list(self):
        v = vec2([1.0, 2.0])
        assert v.shape == (2,)
        assert v[0] == 1.0
        assert v[1] == 2.0
        assert v.dtype == np.float32
    
    def test_vec2_from_tuple(self):
        v = vec2((3.0, 4.0))
        assert v.shape == (2,)
        assert v[0] == 3.0
        assert v[1] == 4.0
    
    def test_vec2_from_array(self):
        arr = np.array([5.0, 6.0])
        v = vec2(arr)
        assert v.shape == (2,)
        assert v[0] == 5.0
        assert v[1] == 6.0


class TestLength2:
    """Test length2 function (2D vector magnitude)"""
    
    def test_length2_unit_x(self):
        v = np.array([1.0, 0.0])
        assert abs(length2(v) - 1.0) < 1e-6
    
    def test_length2_unit_y(self):
        v = np.array([0.0, 1.0])
        assert abs(length2(v) - 1.0) < 1e-6
    
    def test_length2_pythagorean(self):
        v = np.array([3.0, 4.0])
        assert abs(length2(v) - 5.0) < 1e-6
    
    def test_length2_zero(self):
        v = np.array([0.0, 0.0])
        assert abs(length2(v) - 0.0) < 1e-6
    
    def test_length2_negative(self):
        v = np.array([-3.0, -4.0])
        assert abs(length2(v) - 5.0) < 1e-6


class TestNormalize2:
    """Test normalize2 function"""
    
    def test_normalize2_unit_vector(self):
        v = np.array([1.0, 0.0])
        n = normalize2(v)
        assert abs(np.linalg.norm(n) - 1.0) < 1e-6
        assert np.allclose(n, [1.0, 0.0])
    
    def test_normalize2_arbitrary(self):
        v = np.array([3.0, 4.0])
        n = normalize2(v)
        assert abs(np.linalg.norm(n) - 1.0) < 1e-6
        assert np.allclose(n, [0.6, 0.8])
    
    def test_normalize2_zero_vector(self):
        v = np.array([0.0, 0.0])
        n = normalize2(v)
        # Should return zero vector (no crash)
        assert np.allclose(n, [0.0, 0.0])
    
    def test_normalize2_very_small(self):
        v = np.array([1e-15, 1e-15])
        n = normalize2(v)
        # Should handle very small vectors gracefully
        assert not np.any(np.isnan(n))


class TestPerp:
    """Test perp function (perpendicular vector)"""
    
    def test_perp_x_axis(self):
        v = np.array([1.0, 0.0], dtype=np.float32)
        p = perp(v)
        assert np.allclose(p, [0.0, -1.0])
    
    def test_perp_y_axis(self):
        v = np.array([0.0, 1.0], dtype=np.float32)
        p = perp(v)
        assert np.allclose(p, [1.0, 0.0])
    
    def test_perp_arbitrary(self):
        v = np.array([3.0, 4.0], dtype=np.float32)
        p = perp(v)
        assert np.allclose(p, [4.0, -3.0])
    
    def test_perp_is_perpendicular(self):
        v = np.array([5.0, 7.0], dtype=np.float32)
        p = perp(v)
        # Dot product should be zero
        dot = np.dot(v, p)
        assert abs(dot) < 1e-5
    
    def test_perp_preserves_length(self):
        v = np.array([3.0, 4.0], dtype=np.float32)
        p = perp(v)
        assert abs(np.linalg.norm(v) - np.linalg.norm(p)) < 1e-5


class TestBarycentricCoords:
    """Test barycentric_coords function"""
    
    def test_barycentric_at_vertex_a(self):
        a = np.array([0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0], dtype=np.float32)
        c = np.array([0.0, 1.0], dtype=np.float32)
        p = np.array([0.0, 0.0], dtype=np.float32)
        
        bary = barycentric_coords(p, a, b, c)
        assert np.allclose(bary, [1.0, 0.0, 0.0], atol=1e-5)
    
    def test_barycentric_at_vertex_b(self):
        a = np.array([0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0], dtype=np.float32)
        c = np.array([0.0, 1.0], dtype=np.float32)
        p = np.array([1.0, 0.0], dtype=np.float32)
        
        bary = barycentric_coords(p, a, b, c)
        assert np.allclose(bary, [0.0, 1.0, 0.0], atol=1e-5)
    
    def test_barycentric_at_vertex_c(self):
        a = np.array([0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0], dtype=np.float32)
        c = np.array([0.0, 1.0], dtype=np.float32)
        p = np.array([0.0, 1.0], dtype=np.float32)
        
        bary = barycentric_coords(p, a, b, c)
        assert np.allclose(bary, [0.0, 0.0, 1.0], atol=1e-5)
    
    def test_barycentric_at_center(self):
        a = np.array([0.0, 0.0], dtype=np.float32)
        b = np.array([3.0, 0.0], dtype=np.float32)
        c = np.array([0.0, 3.0], dtype=np.float32)
        p = np.array([1.0, 1.0], dtype=np.float32)
        
        bary = barycentric_coords(p, a, b, c)
        # Should sum to 1
        assert abs(np.sum(bary) - 1.0) < 1e-5
        # All should be positive (inside triangle)
        assert np.all(bary >= -1e-5)
    
    def test_barycentric_on_edge(self):
        a = np.array([0.0, 0.0], dtype=np.float32)
        b = np.array([2.0, 0.0], dtype=np.float32)
        c = np.array([0.0, 2.0], dtype=np.float32)
        p = np.array([1.0, 0.0], dtype=np.float32)  # Midpoint of AB
        
        bary = barycentric_coords(p, a, b, c)
        assert abs(np.sum(bary) - 1.0) < 1e-5
        assert abs(bary[2]) < 1e-5  # Should be on edge AB (c weight = 0)
    
    def test_barycentric_degenerate_triangle(self):
        # Collinear points
        a = np.array([0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0], dtype=np.float32)
        c = np.array([2.0, 0.0], dtype=np.float32)
        p = np.array([0.5, 0.0], dtype=np.float32)
        
        bary = barycentric_coords(p, a, b, c)
        # Should return fallback value (1, 0, 0)
        assert bary[0] == 1.0
        assert not np.any(np.isnan(bary))
    
    def test_barycentric_outside_triangle(self):
        a = np.array([0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0], dtype=np.float32)
        c = np.array([0.0, 1.0], dtype=np.float32)
        p = np.array([2.0, 2.0], dtype=np.float32)  # Far outside
        
        bary = barycentric_coords(p, a, b, c)
        # Should still sum to 1
        assert abs(np.sum(bary) - 1.0) < 1e-5
        # At least one coordinate should be negative
        assert np.any(bary < 0.0)
