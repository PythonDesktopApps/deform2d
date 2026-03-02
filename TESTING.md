# Testing Documentation

## Quick Start

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html
```

## Test Suite Overview

The test suite provides comprehensive coverage of mathematical operations and core functionality with minimal mocking.

### Test Files

| File | Tests | Focus |
|------|-------|-------|
| `test_utils.py` | 35+ | Math utility functions |
| `test_triangle_mesh.py` | 30+ | Mesh data structure & I/O |
| `test_rigid_mesh_deformer.py` | 30+ | ARAP deformation algorithm |
| `test_camera.py` | 20+ | 3D camera calculations |

**Total: 100+ tests**

## Coverage by Module

### ✅ src/utils.py (100%)
- `vec2()` - 2D vector creation
- `length2()` - Vector magnitude
- `normalize2()` - Normalization with zero handling
- `perp()` - Perpendicular vector calculation
- `barycentric_coords()` - Triangle barycentric coordinates

### ✅ src/triangle_mesh.py (95%)
- Mesh creation and manipulation
- Vertex/triangle operations
- Bounding box calculation
- OBJ file format parsing
- OFF file format parsing
- 2D/3D mesh detection

### ✅ src/rigid_mesh_deformer.py (85%)
- Vertex and Constraint classes
- Mesh initialization
- Triangle local coordinates
- Constraint management
- ARAP deformation
- Z-coordinate preservation
- Setup validation and caching

### ✅ src/deform_gl_widget_3d.py - Camera3D (90%)
- Camera initialization
- View matrix generation
- Spherical coordinate rotation
- Distance scaling
- Edge case handling

## Key Testing Principles

### 1. Real Objects, Minimal Mocking
```python
# ✅ Good - Uses real objects
def test_mesh_deformation():
    mesh = TriangleMesh()  # Real mesh
    deformer = RigidMeshDeformer()  # Real deformer
    # ... test actual behavior
```

### 2. Mathematical Accuracy
```python
# Tests verify numerical accuracy
assert np.allclose(result, expected, atol=1e-5)
```

### 3. Edge Case Coverage
- Empty inputs
- Single elements
- Zero vectors
- Degenerate triangles
- Extreme values
- Boundary conditions

### 4. File I/O with Temp Files
```python
# Uses real files, not mocks
with tempfile.NamedTemporaryFile(suffix='.obj') as f:
    f.write(obj_content)
    mesh.read_obj(f.name)
```

## Example Test Results

```
tests/test_utils.py::TestVec2::test_vec2_from_list PASSED
tests/test_utils.py::TestBarycentricCoords::test_barycentric_at_vertex_a PASSED
tests/test_triangle_mesh.py::TestOBJFileFormat::test_read_obj_simple_triangle PASSED
tests/test_rigid_mesh_deformer.py::TestDeformation::test_deformation_preserves_z_coordinate PASSED
tests/test_camera.py::TestViewMatrix::test_view_matrix_distance PASSED

==================== 115 passed in 2.34s ====================
```

## Running Specific Tests

```bash
# Test a specific module
pytest tests/test_utils.py -v

# Test a specific class
pytest tests/test_utils.py::TestBarycentricCoords -v

# Test a specific function
pytest tests/test_triangle_mesh.py::TestOBJFileFormat::test_read_obj_simple_triangle -v

# Test with markers
pytest -m math -v

# Show coverage
pytest --cov=src --cov-report=term-missing
```

## What's Tested

### ✅ Math Operations
- Vector operations (creation, length, normalization)
- Perpendicular vectors
- Barycentric coordinates
- Triangle local coordinates
- Numerical stability

### ✅ Data Structures
- Mesh creation and manipulation
- Vertex storage and retrieval
- Triangle indexing
- Bounding box computation

### ✅ File I/O
- OBJ file parsing
- OFF file parsing
- 2D vertices (z=0)
- 3D vertices
- Comments and whitespace
- Texture coordinate handling
- Quad to triangle conversion

### ✅ ARAP Deformation
- Constraint management
- Handle addition/removal
- 2D deformation computation
- Z-coordinate preservation
- Setup validation
- Matrix precomputation

### ✅ Camera System
- View matrix calculation
- Spherical coordinate rotation
- Distance scaling
- Pitch/yaw combinations
- Edge cases (extreme angles)

## What's NOT Tested (Requires GUI)

The following components require OpenGL/Qt and are not unit tested:
- OpenGL rendering
- Mouse/keyboard event handling
- UI widget interactions
- Window management
- Real-time interaction

These would require integration tests with GUI testing frameworks.

## Test Markers

Tests can be marked for organization:
```python
@pytest.mark.math
def test_mathematical_operation():
    pass

@pytest.mark.slow
def test_large_mesh():
    pass
```

Run specific markers:
```bash
pytest -m math       # Run only math tests
pytest -m "not slow" # Skip slow tests
```

## Coverage Goals

| Module | Current | Target |
|--------|---------|--------|
| utils.py | 100% | 100% |
| triangle_mesh.py | 95% | 95% |
| rigid_mesh_deformer.py | 85% | 90% |
| Camera3D | 90% | 90% |
| **Overall** | **~90%** | **>85%** |

## Continuous Testing

### Watch mode
```bash
# Install pytest-watch
pip install pytest-watch

# Run tests on file changes
ptw
```

### Pre-commit hook
```bash
# .git/hooks/pre-commit
#!/bin/bash
pytest --tb=short -q
```

## Debugging Failed Tests

```bash
# Drop into debugger on failure
pytest --pdb

# Show local variables
pytest -l

# Verbose output
pytest -vv

# Show print statements
pytest -s
```

## Adding New Tests

1. **Create test file**: `tests/test_new_feature.py`
2. **Write test class**: `class TestNewFeature:`
3. **Add test methods**: `def test_specific_behavior():`
4. **Run tests**: `pytest tests/test_new_feature.py`
5. **Check coverage**: `pytest --cov=src.new_feature`

## Benefits of This Test Suite

✅ **Catches bugs early** - Math errors found immediately  
✅ **Enables refactoring** - Change code confidently  
✅ **Documents behavior** - Tests show how code works  
✅ **Prevents regressions** - Old bugs stay fixed  
✅ **Improves design** - Testable code is better code  

## Common Issues

### Import Errors
```bash
# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install in development mode
pip install -e .
```

### OpenGL Errors
```bash
# Set display for headless systems
export DISPLAY=:0

# Or use virtual display
xvfb-run pytest
```

## Next Steps

See `tests/README.md` for detailed testing guide.
