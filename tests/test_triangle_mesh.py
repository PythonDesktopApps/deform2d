"""
Unit tests for TriangleMesh class
"""
import numpy as np
import pytest
import tempfile
import os
from src.triangle_mesh import TriangleMesh


class TestTriangleMeshBasics:
    """Test basic TriangleMesh operations"""
    
    def test_init_empty(self):
        mesh = TriangleMesh()
        assert mesh.get_num_vertices() == 0
        assert mesh.get_num_triangles() == 0
    
    def test_append_vertex(self):
        mesh = TriangleMesh()
        mesh.append_vertex([1.0, 2.0, 3.0])
        assert mesh.get_num_vertices() == 1
        v = mesh.get_vertex(0)
        assert np.allclose(v, [1.0, 2.0, 3.0])
    
    def test_append_multiple_vertices(self):
        mesh = TriangleMesh()
        mesh.append_vertex([1.0, 2.0, 3.0])
        mesh.append_vertex([4.0, 5.0, 6.0])
        mesh.append_vertex([7.0, 8.0, 9.0])
        assert mesh.get_num_vertices() == 3
    
    def test_append_triangle(self):
        mesh = TriangleMesh()
        mesh.append_vertex([0.0, 0.0, 0.0])
        mesh.append_vertex([1.0, 0.0, 0.0])
        mesh.append_vertex([0.0, 1.0, 0.0])
        mesh.append_triangle([0, 1, 2])
        assert mesh.get_num_triangles() == 1
    
    def test_get_triangle_indices(self):
        mesh = TriangleMesh()
        mesh.append_vertex([0.0, 0.0, 0.0])
        mesh.append_vertex([1.0, 0.0, 0.0])
        mesh.append_vertex([0.0, 1.0, 0.0])
        mesh.append_triangle([0, 1, 2])
        
        indices = mesh.get_triangle_indices(0)
        assert np.array_equal(indices, [0, 1, 2])
    
    def test_get_triangle_vertices(self):
        mesh = TriangleMesh()
        mesh.append_vertex([0.0, 0.0, 0.0])
        mesh.append_vertex([1.0, 0.0, 0.0])
        mesh.append_vertex([0.0, 1.0, 0.0])
        mesh.append_triangle([0, 1, 2])
        
        verts = mesh.get_triangle_vertices(0)
        assert verts.shape == (3, 3)
        assert np.allclose(verts[0], [0.0, 0.0, 0.0])
        assert np.allclose(verts[1], [1.0, 0.0, 0.0])
        assert np.allclose(verts[2], [0.0, 1.0, 0.0])
    
    def test_set_vertex(self):
        mesh = TriangleMesh()
        mesh.append_vertex([1.0, 2.0, 3.0])
        mesh.set_vertex(0, np.array([4.0, 5.0, 6.0]))
        v = mesh.get_vertex(0)
        assert np.allclose(v, [4.0, 5.0, 6.0])
    
    def test_clear(self):
        mesh = TriangleMesh()
        mesh.append_vertex([1.0, 2.0, 3.0])
        mesh.append_vertex([4.0, 5.0, 6.0])
        mesh.append_triangle([0, 1, 0])
        
        mesh.clear()
        assert mesh.get_num_vertices() == 0
        assert mesh.get_num_triangles() == 0


class TestBoundingBox:
    """Test bounding box calculations"""
    
    def test_bounding_box_empty(self):
        mesh = TriangleMesh()
        bounds = mesh.get_bounding_box()
        assert np.allclose(bounds, [0, 0, 0, 0, 0, 0])
    
    def test_bounding_box_single_vertex(self):
        mesh = TriangleMesh()
        mesh.append_vertex([1.0, 2.0, 3.0])
        bounds = mesh.get_bounding_box()
        assert np.allclose(bounds, [1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    
    def test_bounding_box_cube(self):
        mesh = TriangleMesh()
        mesh.append_vertex([0.0, 0.0, 0.0])
        mesh.append_vertex([1.0, 0.0, 0.0])
        mesh.append_vertex([0.0, 1.0, 0.0])
        mesh.append_vertex([0.0, 0.0, 1.0])
        mesh.append_vertex([1.0, 1.0, 1.0])
        
        bounds = mesh.get_bounding_box()
        assert np.allclose(bounds, [0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    
    def test_bounding_box_negative(self):
        mesh = TriangleMesh()
        mesh.append_vertex([-1.0, -2.0, -3.0])
        mesh.append_vertex([1.0, 2.0, 3.0])
        
        bounds = mesh.get_bounding_box()
        assert np.allclose(bounds, [-1.0, 1.0, -2.0, 2.0, -3.0, 3.0])


class TestOBJFileFormat:
    """Test OBJ file loading"""
    
    def test_read_obj_simple_triangle(self):
        # Create temporary OBJ file
        obj_content = """# Simple triangle
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
f 1 2 3
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as f:
            f.write(obj_content)
            temp_path = f.name
        
        try:
            mesh = TriangleMesh()
            mesh.read_obj(temp_path)
            
            assert mesh.get_num_vertices() == 3
            assert mesh.get_num_triangles() == 1
            assert np.allclose(mesh.get_vertex(0), [0.0, 0.0, 0.0])
            assert np.allclose(mesh.get_vertex(1), [1.0, 0.0, 0.0])
            assert np.allclose(mesh.get_vertex(2), [0.0, 1.0, 0.0])
        finally:
            os.unlink(temp_path)
    
    def test_read_obj_with_comments(self):
        obj_content = """# This is a comment
# Another comment
v 0.0 0.0 0.0
# More comments
v 1.0 0.0 0.0
v 0.0 1.0 0.0
f 1 2 3
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as f:
            f.write(obj_content)
            temp_path = f.name
        
        try:
            mesh = TriangleMesh()
            mesh.read_obj(temp_path)
            assert mesh.get_num_vertices() == 3
            assert mesh.get_num_triangles() == 1
        finally:
            os.unlink(temp_path)
    
    def test_read_obj_2d_vertices(self):
        # 2D vertices (no Z coordinate)
        obj_content = """v 0.0 0.0
v 1.0 0.0
v 0.0 1.0
f 1 2 3
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as f:
            f.write(obj_content)
            temp_path = f.name
        
        try:
            mesh = TriangleMesh()
            mesh.read_obj(temp_path)
            
            assert mesh.get_num_vertices() == 3
            # Z should be 0
            assert mesh.get_vertex(0)[2] == 0.0
            assert mesh.get_vertex(1)[2] == 0.0
            assert mesh.get_vertex(2)[2] == 0.0
        finally:
            os.unlink(temp_path)
    
    def test_read_obj_with_texture_coords(self):
        # Face with texture coordinates (should be ignored)
        obj_content = """v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
vt 0.0 0.0
vt 1.0 0.0
vt 0.0 1.0
f 1/1 2/2 3/3
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as f:
            f.write(obj_content)
            temp_path = f.name
        
        try:
            mesh = TriangleMesh()
            mesh.read_obj(temp_path)
            
            assert mesh.get_num_vertices() == 3
            assert mesh.get_num_triangles() == 1
            assert np.array_equal(mesh.get_triangle_indices(0), [0, 1, 2])
        finally:
            os.unlink(temp_path)


class TestOFFFileFormat:
    """Test OFF file loading"""
    
    def test_read_off_simple_triangle(self):
        off_content = """OFF
3 1 0
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
3 0 1 2
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.off', delete=False) as f:
            f.write(off_content)
            temp_path = f.name
        
        try:
            mesh = TriangleMesh()
            mesh.read_off(temp_path)
            
            assert mesh.get_num_vertices() == 3
            assert mesh.get_num_triangles() == 1
            assert np.allclose(mesh.get_vertex(0), [0.0, 0.0, 0.0])
            assert np.allclose(mesh.get_vertex(1), [1.0, 0.0, 0.0])
            assert np.allclose(mesh.get_vertex(2), [0.0, 1.0, 0.0])
        finally:
            os.unlink(temp_path)
    
    def test_read_off_quad_splits(self):
        # Quad should be split into two triangles
        off_content = """OFF
4 1 0
0.0 0.0 0.0
1.0 0.0 0.0
1.0 1.0 0.0
0.0 1.0 0.0
4 0 1 2 3
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.off', delete=False) as f:
            f.write(off_content)
            temp_path = f.name
        
        try:
            mesh = TriangleMesh()
            mesh.read_off(temp_path)
            
            assert mesh.get_num_vertices() == 4
            assert mesh.get_num_triangles() == 2  # Quad split into 2 triangles
        finally:
            os.unlink(temp_path)
    
    def test_read_off_multiple_triangles(self):
        off_content = """OFF
4 2 0
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
1.0 1.0 0.0
3 0 1 2
3 1 3 2
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.off', delete=False) as f:
            f.write(off_content)
            temp_path = f.name
        
        try:
            mesh = TriangleMesh()
            mesh.read_off(temp_path)
            
            assert mesh.get_num_vertices() == 4
            assert mesh.get_num_triangles() == 2
        finally:
            os.unlink(temp_path)
    
    def test_read_off_with_comments(self):
        off_content = """OFF
# This is a comment
3 1 0
# Comment in vertices
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
# Comment in faces
3 0 1 2
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.off', delete=False) as f:
            f.write(off_content)
            temp_path = f.name
        
        try:
            mesh = TriangleMesh()
            mesh.read_off(temp_path)
            
            assert mesh.get_num_vertices() == 3
            assert mesh.get_num_triangles() == 1
        finally:
            os.unlink(temp_path)
    
    def test_read_off_invalid_header(self):
        off_content = """INVALID
3 1 0
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
3 0 1 2
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.off', delete=False) as f:
            f.write(off_content)
            temp_path = f.name
        
        try:
            mesh = TriangleMesh()
            with pytest.raises(ValueError):
                mesh.read_off(temp_path)
        finally:
            os.unlink(temp_path)


class TestMeshOperations:
    """Test mesh manipulation operations"""
    
    def test_copy_vertex(self):
        mesh = TriangleMesh()
        mesh.append_vertex([1.0, 2.0, 3.0])
        v1 = mesh.get_vertex(0)
        v2 = mesh.get_vertex(0)
        
        # Should be copies, not same object
        v1[0] = 99.0
        assert v2[0] != 99.0
    
    def test_multiple_triangles_same_vertices(self):
        mesh = TriangleMesh()
        mesh.append_vertex([0.0, 0.0, 0.0])
        mesh.append_vertex([1.0, 0.0, 0.0])
        mesh.append_vertex([0.0, 1.0, 0.0])
        mesh.append_vertex([1.0, 1.0, 0.0])
        
        mesh.append_triangle([0, 1, 2])
        mesh.append_triangle([1, 3, 2])
        
        assert mesh.get_num_vertices() == 4
        assert mesh.get_num_triangles() == 2
