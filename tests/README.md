# Test Suite for ARAP Mesh Deformation

This directory contains comprehensive unit tests for the As-Rigid-As-Possible mesh deformation application.

## Running Tests

### Run all tests
```bash
pytest
```

### Run with verbose output
```bash
pytest -v
```

### Run specific test file
```bash
pytest tests/test_utils.py
pytest tests/test_triangle_mesh.py
pytest tests/test_rigid_mesh_deformer.py
pytest tests/test_camera.py
```

### Run specific test class
```bash
pytest tests/test_utils.py::TestBarycentricCoords
```

### Run specific test
```bash
pytest tests/test_utils.py::TestBarycentricCoords::test_barycentric_at_vertex_a
```

### Run with coverage report
```bash
pytest --cov=src --cov-report=html --cov-report=term
```

This will create an HTML coverage report in `htmlcov/index.html`.

### Run only fast tests (exclude slow tests)
```bash
pytest -m "not slow"
```

### Run only math tests
```bash
pytest -m math
```

## Test Organization

### test_utils.py
Tests for utility math functions:
- `vec2()` - 2D vector creation
- `length2()` - Vector magnitude
- `normalize2()` - Vector normalization
- `perp()` - Perpendicular vector
- `barycentric_coords()` - Barycentric coordinate calculation

**Coverage:**
- ✅ Basic functionality
- ✅ Edge cases (zero vectors, degenerate cases)
- ✅ Numerical accuracy
- ✅ Type handling

### test_triangle_mesh.py
Tests for TriangleMesh class:
- Mesh creation and manipulation
- Vertex and triangle operations
- Bounding box calculations
- OBJ file loading
- OFF file loading

**Coverage:**
- ✅ Empty mesh handling
- ✅ Single and multiple vertices/triangles
- ✅ File format parsing (OBJ, OFF)
- ✅ Comments and whitespace handling
- ✅ 2D vs 3D mesh detection
- ✅ Quad to triangle conversion
- ✅ Invalid file format handling

### test_rigid_mesh_deformer.py
Tests for RigidMeshDeformer (ARAP algorithm):
- Vertex and Constraint classes
- Mesh initialization
- Constraint management
- Deformation calculations
- Setup validation

**Coverage:**
- ✅ 2D and 3D vertex support
- ✅ Constraint equality and hashing
- ✅ Deformer initialization from mesh
- ✅ Triangle coordinate computation
- ✅ Handle addition/removal
- ✅ Z-coordinate preservation
- ✅ Setup validation and caching
- ✅ Edge cases (insufficient constraints, empty mesh)

### test_camera.py
Tests for Camera3D class:
- Camera initialization
- View matrix calculation
- Rotation (pitch/yaw)
- Distance and positioning

**Coverage:**
- ✅ Basic camera setup
- ✅ View matrix generation
- ✅ Rotation calculations
- ✅ Distance scaling
- ✅ Edge cases (extreme angles, zero distance)
- ✅ Mathematical accuracy

## Test Philosophy

### No Mocking (Minimal Mocking)
Tests use real objects and data wherever possible:
- ✅ Real TriangleMesh objects
- ✅ Real RigidMeshDeformer instances
- ✅ Actual file I/O with temporary files
- ✅ Real numpy arrays and calculations

**Why?** This ensures tests catch real integration issues and verify actual behavior.

### Mathematical Accuracy
Tests verify numerical accuracy:
- Tolerance levels appropriate for float32
- Edge case handling (division by zero, degenerate triangles)
- Numerical stability checks

### Comprehensive Coverage
Tests cover:
- ✅ Happy paths (normal usage)
- ✅ Edge cases (empty, single element)
- ✅ Error cases (invalid input)
- ✅ Boundary conditions (extreme values)

## Test Statistics

**Total Tests:** 100+

**Test Files:** 4
- test_utils.py: ~35 tests
- test_triangle_mesh.py: ~30 tests
- test_rigid_mesh_deformer.py: ~30 tests
- test_camera.py: ~20 tests

**Code Coverage Target:** >80%

## Common Issues

### Import Errors
If you get import errors:
```bash
# Make sure you're in the project root
cd /path/to/deform2d

# Install in development mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### OpenGL/PyQt5 Dependencies
Some tests may require display/OpenGL:
```bash
# On Linux, you may need
export DISPLAY=:0

# Or use Xvfb for headless testing
xvfb-run pytest
```

### Temporary Files
Tests use `tempfile` module which handles cleanup automatically.
If tests are interrupted, check `/tmp` for leftover test files.

## Writing New Tests

### Test Template
```python
import pytest
import numpy as np
from src.your_module import YourClass

class TestYourFeature:
    """Test description"""
    
    def test_basic_case(self):
        """Test basic functionality"""
        obj = YourClass()
        result = obj.method()
        assert result == expected
    
    def test_edge_case(self):
        """Test edge case"""
        obj = YourClass()
        # Test edge case
        assert ...
    
    def test_error_handling(self):
        """Test error handling"""
        obj = YourClass()
        with pytest.raises(ValueError):
            obj.method(invalid_input)
```

### Test Naming Convention
- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`

### Assertions
Use pytest's assert with informative messages:
```python
assert result == expected, f"Expected {expected}, got {result}"
assert np.allclose(a, b, atol=1e-5), "Arrays not close enough"
```

### Fixtures
For commonly used objects:
```python
@pytest.fixture
def simple_mesh():
    mesh = TriangleMesh()
    mesh.append_vertex([0, 0, 0])
    mesh.append_vertex([1, 0, 0])
    mesh.append_vertex([0, 1, 0])
    mesh.append_triangle([0, 1, 2])
    return mesh

def test_with_fixture(simple_mesh):
    assert simple_mesh.get_num_vertices() == 3
```

## Continuous Integration

To set up CI (GitHub Actions, GitLab CI, etc.):

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: pytest --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Debugging Tests

### Run with pdb on failure
```bash
pytest --pdb
```

### Show print statements
```bash
pytest -s
```

### Show local variables on failure
```bash
pytest -l
```

### Run last failed tests only
```bash
pytest --lf
```

### Run failed tests first
```bash
pytest --ff
```

## Performance

Tests should run quickly:
- Full suite: <10 seconds
- Individual files: <3 seconds

Mark slow tests:
```python
@pytest.mark.slow
def test_large_mesh_deformation():
    # Test with 10k vertices
    pass
```

## Contributing

When adding new features:
1. Write tests first (TDD)
2. Ensure >80% coverage
3. Test edge cases
4. Run full test suite before committing
5. Update this README if needed

## License

Same as main project.
