"""
Unit tests for RigidMeshDeformer (ARAP deformation)
"""
import numpy as np
import pytest
from src.triangle_mesh import TriangleMesh
from src.rigid_mesh_deformer import RigidMeshDeformer, Vertex, Triangle, Constraint


class TestVertex:
    """Test Vertex class"""
    
    def test_vertex_2d(self):
        v = Vertex([1.0, 2.0])
        assert v.vPosition.shape == (3,)
        assert v.vPosition[0] == 1.0
        assert v.vPosition[1] == 2.0
        assert v.vPosition[2] == 0.0  # Z should be 0 for 2D
    
    def test_vertex_3d(self):
        v = Vertex([1.0, 2.0, 3.0])
        assert v.vPosition.shape == (3,)
        assert v.vPosition[0] == 1.0
        assert v.vPosition[1] == 2.0
        assert v.vPosition[2] == 3.0


class TestConstraint:
    """Test Constraint class"""
    
    def test_constraint_2d(self):
        c = Constraint(5, [1.0, 2.0])
        assert c.nVertex == 5
        assert c.vConstrainedPos.shape == (3,)
        assert c.vConstrainedPos[0] == 1.0
        assert c.vConstrainedPos[1] == 2.0
        assert c.vConstrainedPos[2] == 0.0
    
    def test_constraint_3d(self):
        c = Constraint(3, [1.0, 2.0, 3.0])
        assert c.nVertex == 3
        assert c.vConstrainedPos[0] == 1.0
        assert c.vConstrainedPos[1] == 2.0
        assert c.vConstrainedPos[2] == 3.0
    
    def test_constraint_equality(self):
        c1 = Constraint(5, [1.0, 2.0])
        c2 = Constraint(5, [3.0, 4.0])
        c3 = Constraint(7, [1.0, 2.0])
        
        assert c1 == c2  # Same vertex index
        assert c1 != c3  # Different vertex index
    
    def test_constraint_hash(self):
        c1 = Constraint(5, [1.0, 2.0])
        c2 = Constraint(5, [3.0, 4.0])
        
        # Should have same hash (based on vertex index)
        assert hash(c1) == hash(c2)
        
        # Can be used in sets
        s = {c1, c2}
        assert len(s) == 1
    
    def test_constraint_ordering(self):
        c1 = Constraint(3, [0.0, 0.0])
        c2 = Constraint(7, [0.0, 0.0])
        c3 = Constraint(1, [0.0, 0.0])
        
        assert c3 < c1 < c2
        sorted_list = sorted([c2, c1, c3])
        assert sorted_list[0].nVertex == 1
        assert sorted_list[1].nVertex == 3
        assert sorted_list[2].nVertex == 7


class TestDeformerInitialization:
    """Test deformer initialization"""
    
    def test_init_empty(self):
        deformer = RigidMeshDeformer()
        assert len(deformer.m_vConstraints) == 0
        assert len(deformer.m_vInitialVerts) == 0
        assert len(deformer.m_vDeformedVerts) == 0
        assert len(deformer.m_vTriangles) == 0
        assert not deformer.m_bSetupValid
    
    def test_initialize_from_simple_triangle(self):
        mesh = TriangleMesh()
        mesh.append_vertex([0.0, 0.0, 0.0])
        mesh.append_vertex([1.0, 0.0, 0.0])
        mesh.append_vertex([0.0, 1.0, 0.0])
        mesh.append_triangle([0, 1, 2])
        
        deformer = RigidMeshDeformer()
        deformer.initialize_from_mesh(mesh)
        
        assert len(deformer.m_vInitialVerts) == 3
        assert len(deformer.m_vDeformedVerts) == 3
        assert len(deformer.m_vTriangles) == 1
    
    def test_initialize_preserves_vertices(self):
        mesh = TriangleMesh()
        mesh.append_vertex([1.0, 2.0, 3.0])
        mesh.append_vertex([4.0, 5.0, 6.0])
        mesh.append_vertex([7.0, 8.0, 9.0])
        mesh.append_triangle([0, 1, 2])
        
        deformer = RigidMeshDeformer()
        deformer.initialize_from_mesh(mesh)
        
        assert np.allclose(deformer.m_vInitialVerts[0].vPosition, [1.0, 2.0, 3.0])
        assert np.allclose(deformer.m_vInitialVerts[1].vPosition, [4.0, 5.0, 6.0])
        assert np.allclose(deformer.m_vInitialVerts[2].vPosition, [7.0, 8.0, 9.0])
    
    def test_initialize_computes_triangle_coords(self):
        mesh = TriangleMesh()
        mesh.append_vertex([0.0, 0.0, 0.0])
        mesh.append_vertex([1.0, 0.0, 0.0])
        mesh.append_vertex([0.0, 1.0, 0.0])
        mesh.append_triangle([0, 1, 2])
        
        deformer = RigidMeshDeformer()
        deformer.initialize_from_mesh(mesh)
        
        tri = deformer.m_vTriangles[0]
        # Triangle coordinates should be computed
        assert tri.vTriCoords.shape == (3, 2)
        # Not all zeros
        assert not np.allclose(tri.vTriCoords, 0.0)


class TestConstraintManagement:
    """Test constraint handling"""
    
    def test_set_deformed_handle(self):
        mesh = TriangleMesh()
        mesh.append_vertex([0.0, 0.0, 0.0])
        mesh.append_vertex([1.0, 0.0, 0.0])
        mesh.append_vertex([0.0, 1.0, 0.0])
        mesh.append_triangle([0, 1, 2])
        
        deformer = RigidMeshDeformer()
        deformer.initialize_from_mesh(mesh)
        deformer.set_deformed_handle(0, (0.5, 0.5, 0.0))
        
        assert len(deformer.m_vConstraints) == 1
        assert any(c.nVertex == 0 for c in deformer.m_vConstraints)
    
    def test_remove_handle(self):
        mesh = TriangleMesh()
        mesh.append_vertex([0.0, 0.0, 0.0])
        mesh.append_vertex([1.0, 0.0, 0.0])
        mesh.append_vertex([0.0, 1.0, 0.0])
        mesh.append_triangle([0, 1, 2])
        
        deformer = RigidMeshDeformer()
        deformer.initialize_from_mesh(mesh)
        deformer.set_deformed_handle(0, (0.5, 0.5, 0.0))
        
        assert len(deformer.m_vConstraints) == 1
        
        deformer.remove_handle(0)
        assert len(deformer.m_vConstraints) == 0
    
    def test_update_existing_constraint(self):
        mesh = TriangleMesh()
        mesh.append_vertex([0.0, 0.0, 0.0])
        mesh.append_vertex([1.0, 0.0, 0.0])
        mesh.append_vertex([0.0, 1.0, 0.0])
        mesh.append_triangle([0, 1, 2])
        
        deformer = RigidMeshDeformer()
        deformer.initialize_from_mesh(mesh)
        deformer.set_deformed_handle(0, (0.5, 0.5, 0.0))
        deformer.set_deformed_handle(0, (0.7, 0.8, 0.0))
        
        # Should still be just one constraint
        assert len(deformer.m_vConstraints) == 1
        
        # Position should be updated
        constraint = list(deformer.m_vConstraints)[0]
        assert np.allclose(constraint.vConstrainedPos[:2], [0.7, 0.8])


class TestDeformation:
    """Test mesh deformation"""
    
    def test_deformation_with_insufficient_constraints(self):
        """With less than 2 constraints, mesh should stay at initial position"""
        mesh = TriangleMesh()
        mesh.append_vertex([0.0, 0.0, 0.0])
        mesh.append_vertex([1.0, 0.0, 0.0])
        mesh.append_vertex([0.0, 1.0, 0.0])
        mesh.append_triangle([0, 1, 2])
        
        deformer = RigidMeshDeformer()
        deformer.initialize_from_mesh(mesh)
        
        # Set only one constraint (not enough)
        deformer.set_deformed_handle(0, (0.5, 0.5, 0.0))
        
        # Update mesh
        deformed = TriangleMesh()
        for i in range(mesh.get_num_vertices()):
            deformed.append_vertex(mesh.get_vertex(i))
        for i in range(mesh.get_num_triangles()):
            deformed.append_triangle(mesh.get_triangle_indices(i))
        
        deformer.update_deformed_mesh(deformed, bRigid=False)
        
        # Mesh should be unchanged (uses initial verts)
        for i in range(mesh.get_num_vertices()):
            assert np.allclose(deformed.get_vertex(i), mesh.get_vertex(i))
    
    def test_deformation_preserves_z_coordinate(self):
        """Z coordinate should be preserved during 2D deformation"""
        mesh = TriangleMesh()
        mesh.append_vertex([0.0, 0.0, 1.0])  # Z = 1.0
        mesh.append_vertex([1.0, 0.0, 2.0])  # Z = 2.0
        mesh.append_vertex([0.0, 1.0, 3.0])  # Z = 3.0
        mesh.append_triangle([0, 1, 2])
        
        deformer = RigidMeshDeformer()
        deformer.initialize_from_mesh(mesh)
        
        # Set two constraints
        deformer.set_deformed_handle(0, (0.0, 0.0, 1.0))
        deformer.set_deformed_handle(1, (1.5, 0.0, 2.0))
        
        # Update mesh
        deformed = TriangleMesh()
        for i in range(mesh.get_num_vertices()):
            deformed.append_vertex(mesh.get_vertex(i))
        for i in range(mesh.get_num_triangles()):
            deformed.append_triangle(mesh.get_triangle_indices(i))
        
        deformer.update_deformed_mesh(deformed, bRigid=True)
        
        # Z coordinates should be preserved
        assert abs(deformed.get_vertex(0)[2] - 1.0) < 1e-5
        assert abs(deformed.get_vertex(1)[2] - 2.0) < 1e-5
        assert abs(deformed.get_vertex(2)[2] - 3.0) < 1e-5
    
    def test_simple_square_deformation(self):
        """Test deformation on a simple square mesh"""
        mesh = TriangleMesh()
        # Create a simple 2x2 grid
        mesh.append_vertex([0.0, 0.0, 0.0])  # 0
        mesh.append_vertex([1.0, 0.0, 0.0])  # 1
        mesh.append_vertex([0.0, 1.0, 0.0])  # 2
        mesh.append_vertex([1.0, 1.0, 0.0])  # 3
        
        mesh.append_triangle([0, 1, 2])
        mesh.append_triangle([1, 3, 2])
        
        deformer = RigidMeshDeformer()
        deformer.initialize_from_mesh(mesh)
        
        # Pin bottom two corners
        deformer.set_deformed_handle(0, (0.0, 0.0, 0.0))
        deformer.set_deformed_handle(1, (1.0, 0.0, 0.0))
        
        # Deform top-right corner
        deformer.set_deformed_handle(3, (1.5, 1.0, 0.0))
        
        # Update mesh
        deformed = TriangleMesh()
        for i in range(mesh.get_num_vertices()):
            deformed.append_vertex(mesh.get_vertex(i))
        for i in range(mesh.get_num_triangles()):
            deformed.append_triangle(mesh.get_triangle_indices(i))
        
        deformer.update_deformed_mesh(deformed, bRigid=True)
        
        # Pinned vertices should be at exact positions
        assert np.allclose(deformed.get_vertex(0)[:2], [0.0, 0.0], atol=1e-5)
        assert np.allclose(deformed.get_vertex(1)[:2], [1.0, 0.0], atol=1e-5)
        assert np.allclose(deformed.get_vertex(3)[:2], [1.5, 1.0], atol=1e-5)
        
        # Free vertex should have moved
        free_vert = deformed.get_vertex(2)
        # Should not be at original position
        assert not np.allclose(free_vert[:2], [0.0, 1.0], atol=1e-2)


class TestSetupValidation:
    """Test setup validation and caching"""
    
    def test_setup_invalidation(self):
        mesh = TriangleMesh()
        mesh.append_vertex([0.0, 0.0, 0.0])
        mesh.append_vertex([1.0, 0.0, 0.0])
        mesh.append_vertex([0.0, 1.0, 0.0])
        mesh.append_triangle([0, 1, 2])
        
        deformer = RigidMeshDeformer()
        deformer.initialize_from_mesh(mesh)
        
        assert not deformer.m_bSetupValid
        
        # Add constraints
        deformer.set_deformed_handle(0, (0.0, 0.0, 0.0))
        deformer.set_deformed_handle(1, (1.0, 0.0, 0.0))
        
        # Validate
        deformer.validate_setup()
        assert deformer.m_bSetupValid
        
        # Adding new constraint should invalidate
        deformer.set_deformed_handle(2, (0.0, 1.0, 0.0))
        # Setup should still be valid (constraint updated, not added)
        # Actually, adding a NEW constraint invalidates
        
    def test_validate_setup_with_sufficient_constraints(self):
        mesh = TriangleMesh()
        mesh.append_vertex([0.0, 0.0, 0.0])
        mesh.append_vertex([1.0, 0.0, 0.0])
        mesh.append_vertex([0.0, 1.0, 0.0])
        mesh.append_triangle([0, 1, 2])
        
        deformer = RigidMeshDeformer()
        deformer.initialize_from_mesh(mesh)
        
        deformer.set_deformed_handle(0, (0.0, 0.0, 0.0))
        deformer.set_deformed_handle(1, (1.0, 0.0, 0.0))
        
        deformer.validate_setup()
        assert deformer.m_bSetupValid
        assert deformer.m_mFirstMatrix is not None


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_mesh(self):
        mesh = TriangleMesh()
        deformer = RigidMeshDeformer()
        deformer.initialize_from_mesh(mesh)
        
        assert len(deformer.m_vInitialVerts) == 0
        assert len(deformer.m_vTriangles) == 0
    
    def test_single_triangle_minimal_deformation(self):
        mesh = TriangleMesh()
        mesh.append_vertex([0.0, 0.0, 0.0])
        mesh.append_vertex([1.0, 0.0, 0.0])
        mesh.append_vertex([0.0, 1.0, 0.0])
        mesh.append_triangle([0, 1, 2])
        
        deformer = RigidMeshDeformer()
        deformer.initialize_from_mesh(mesh)
        
        # Pin all three vertices at original positions
        deformer.set_deformed_handle(0, (0.0, 0.0, 0.0))
        deformer.set_deformed_handle(1, (1.0, 0.0, 0.0))
        deformer.set_deformed_handle(2, (0.0, 1.0, 0.0))
        
        deformed = TriangleMesh()
        for i in range(mesh.get_num_vertices()):
            deformed.append_vertex(mesh.get_vertex(i))
        for i in range(mesh.get_num_triangles()):
            deformed.append_triangle(mesh.get_triangle_indices(i))
        
        deformer.update_deformed_mesh(deformed, bRigid=True)
        
        # All vertices should be at original positions
        for i in range(3):
            assert np.allclose(deformed.get_vertex(i), mesh.get_vertex(i), atol=1e-5)
