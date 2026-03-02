#!/usr/bin/env python
"""
Test script to verify 3D features work correctly
"""
import numpy as np
from src.triangle_mesh import TriangleMesh
from src.rigid_mesh_deformer import RigidMeshDeformer

def test_off_loading():
    """Test loading OFF file format"""
    print("Testing OFF file loading...")
    mesh = TriangleMesh()
    
    try:
        mesh.read_off("assets/armadillo_250.off")
        print(f"✓ Successfully loaded armadillo mesh")
        print(f"  Vertices: {mesh.get_num_vertices()}")
        print(f"  Triangles: {mesh.get_num_triangles()}")
        
        # Check bounds
        bounds = mesh.get_bounding_box()
        print(f"  Bounds X: [{bounds[0]:.3f}, {bounds[1]:.3f}]")
        print(f"  Bounds Y: [{bounds[2]:.3f}, {bounds[3]:.3f}]")
        print(f"  Bounds Z: [{bounds[4]:.3f}, {bounds[5]:.3f}]")
        
        # Check if truly 3D
        z_range = bounds[5] - bounds[4]
        is_3d = z_range > 1e-6
        print(f"  Is 3D: {is_3d} (Z range: {z_range:.6f})")
        
        return True
    except Exception as e:
        print(f"✗ Failed to load OFF file: {e}")
        return False

def test_3d_deformer():
    """Test deformer with 3D vertices"""
    print("\nTesting 3D deformer...")
    mesh = TriangleMesh()
    
    try:
        mesh.read_off("assets/armadillo_250.off")
        
        deformer = RigidMeshDeformer()
        deformer.initialize_from_mesh(mesh)
        
        print(f"✓ Deformer initialized")
        print(f"  Initial vertices: {len(deformer.m_vInitialVerts)}")
        print(f"  Deformed vertices: {len(deformer.m_vDeformedVerts)}")
        
        # Check first vertex has 3D position
        v0 = deformer.m_vInitialVerts[0].vPosition
        print(f"  First vertex: ({v0[0]:.3f}, {v0[1]:.3f}, {v0[2]:.3f})")
        
        has_z = abs(v0[2]) > 1e-6
        print(f"  Has Z component: {has_z}")
        
        # Test constraint setting
        test_pos = [0.1, 0.2, 0.3]
        deformer.set_deformed_handle(0, test_pos)
        print(f"✓ Set 3D constraint successfully")
        
        return True
    except Exception as e:
        print(f"✗ Failed deformer test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_obj_loading():
    """Test OBJ file still works"""
    print("\nTesting OBJ file loading...")
    mesh = TriangleMesh()
    
    try:
        mesh.read_obj("assets/man.obj")
        print(f"✓ Successfully loaded man.obj")
        print(f"  Vertices: {mesh.get_num_vertices()}")
        print(f"  Triangles: {mesh.get_num_triangles()}")
        
        bounds = mesh.get_bounding_box()
        z_range = bounds[5] - bounds[4]
        is_3d = z_range > 1e-6
        print(f"  Is 3D: {is_3d} (Z range: {z_range:.6f})")
        
        return True
    except Exception as e:
        print(f"✗ Failed to load OBJ file: {e}")
        return False

def test_simple_mesh_creation():
    """Test creating a simple 3D mesh programmatically"""
    print("\nTesting simple 3D mesh creation...")
    mesh = TriangleMesh()
    
    # Create a simple tetrahedron
    mesh.append_vertex([0.0, 0.0, 0.0])
    mesh.append_vertex([1.0, 0.0, 0.0])
    mesh.append_vertex([0.5, 1.0, 0.0])
    mesh.append_vertex([0.5, 0.5, 1.0])
    
    mesh.append_triangle([0, 1, 2])
    mesh.append_triangle([0, 1, 3])
    mesh.append_triangle([1, 2, 3])
    mesh.append_triangle([2, 0, 3])
    
    print(f"✓ Created tetrahedron")
    print(f"  Vertices: {mesh.get_num_vertices()}")
    print(f"  Triangles: {mesh.get_num_triangles()}")
    
    # Test deformer with it
    deformer = RigidMeshDeformer()
    deformer.initialize_from_mesh(mesh)
    print(f"✓ Deformer initialized with 3D tetrahedron")
    
    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("3D Features Test Suite")
    print("=" * 60)
    
    tests = [
        test_off_loading,
        test_3d_deformer,
        test_obj_loading,
        test_simple_mesh_creation
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)
    
    if all(results):
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    exit(main())
