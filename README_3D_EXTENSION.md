# 3D Extension for ARAP Mesh Deformation

This document describes the extensions made to support **3D mesh deformation** in addition to the original 2D functionality.

## Overview

The application has been extended to support **As-Rigid-As-Possible (ARAP) deformation** for both 2D and 3D meshes. The same interaction paradigm is maintained:
- **Right-click** on vertices to pin them (at least 2 required)
- **Left-click and drag** a vertex to deform the mesh
- The mesh deforms in an as-rigid-as-possible manner

## What's New

### 1. 3D Mesh Support
- **OFF file format support**: Can now load `.off` files (common 3D mesh format)
- **3D OBJ support**: Improved OBJ loader to handle true 3D meshes
- **Example mesh**: Included `armadillo_250.off` in the assets folder

### 2. 3D Camera System
- **Trackball rotation**: Left-click and drag (when not on a vertex) to rotate the camera
- **Zoom**: Mouse wheel to zoom in/out
- **Auto-fit**: Camera automatically adjusts to fit loaded meshes
- **Reset**: Press `R` to reset camera to default view

### 3. 3D Interaction
- **Ray-casting**: Vertices are selected using 3D ray-casting from screen to world space
- **Drag plane**: When dragging a vertex, it moves along a plane perpendicular to the camera view
- **Depth-aware picking**: Vertices closer to camera are prioritized when multiple are near the mouse

### 4. Enhanced Visualization
- **Wireframe + Filled**: Meshes are rendered with both wireframe edges and filled triangles
- **Lighting**: Basic lighting system for better 3D perception
- **Depth testing**: Proper depth buffering for 3D rendering
- **Highlighted vertices**: Pinned vertices shown in red

## File Changes

### New Files
- **`src/deform_gl_widget_3d.py`**: New 3D-capable OpenGL widget
  - `Camera3D`: Trackball-style 3D camera
  - `DeformGLWidget3D`: Main widget with 3D support
  - Ray-casting for vertex selection
  - Drag plane for 3D vertex manipulation

### Modified Files
- **`src/triangle_mesh.py`**:
  - Added `read_off()` method for OFF file format support
  
- **`src/rigid_mesh_deformer.py`**:
  - Updated `Vertex` class to support 3D positions
  - Updated `Constraint` class to support 3D positions
  - Modified deformation to preserve Z coordinates
  
- **`main.py`**:
  - Now uses `DeformGLWidget3D` instead of `DeformGLWidget`
  - Added status bar with keyboard/mouse instructions

## Usage

### Running the Application
```bash
python main.py
```

The application will load the armadillo mesh by default (if available), otherwise it falls back to the 2D man mesh.

### Controls

#### Mouse Controls
- **Right-click on vertex**: Pin/unpin vertex (creates constraint)
  - You need at least 2 pinned vertices to deform
  - Right-click again on a pinned vertex to unpin it
  
- **Left-click on vertex + drag**: Deform the mesh
  - Click and hold on any vertex
  - Drag to move it in 3D space
  - The mesh deforms rigidly to accommodate the movement
  
- **Left-click on empty space + drag**: Rotate camera (3D mode only)
  - Horizontal movement: rotate around Y-axis (yaw)
  - Vertical movement: rotate around X-axis (pitch)
  
- **Mouse wheel**: Zoom in/out

#### Keyboard Controls
- **F**: Open file dialog to load a mesh (.obj or .off)
- **R**: Reset camera to fit mesh
- **C**: Clear all constraints (unpin all vertices)

### Workflow Example

1. **Load a 3D mesh**: 
   - Press `F` and select `assets/armadillo_250.off`
   
2. **Rotate to desired view**:
   - Left-click and drag in empty space to rotate
   - Use mouse wheel to zoom
   
3. **Pin control vertices**:
   - Right-click on 2 or more vertices to pin them
   - These vertices will stay in place during deformation
   
4. **Deform the mesh**:
   - Left-click on any vertex (pinned or unpinned)
   - Drag it to deform the mesh
   - The mesh will deform as-rigidly-as-possible
   
5. **Adjust**:
   - Add more pins by right-clicking more vertices
   - Remove pins by right-clicking pinned vertices again
   - Clear all with `C` key

## Technical Details

### ARAP Deformation in 3D

The deformation algorithm works in **projected 2D space** but maintains **3D positions**:

1. **Vertices are stored in 3D** (x, y, z coordinates)
2. **Deformation is computed in 2D** (x, y only) using the original ARAP algorithm
3. **Z-coordinates are preserved** during deformation
4. **Dragging uses a view-aligned plane** in 3D space

This approach works well for:
- ✅ Surface meshes with shallow Z variation
- ✅ Character meshes viewed from one side
- ✅ Meshes where deformation is primarily in the XY plane

For fully 3D volumetric deformation, you would need to extend the ARAP solver to work in true 3D (which is significantly more complex).

### OFF File Format

The OFF (Object File Format) parser handles:
- Triangle meshes (3 vertices per face)
- Quad meshes (automatically split into 2 triangles)
- Mixed meshes with varying face sizes

Format example:
```
OFF
<num_vertices> <num_faces> 0
<x> <y> <z>    # vertex 0
<x> <y> <z>    # vertex 1
...
3 <v0> <v1> <v2>    # triangle face
4 <v0> <v1> <v2> <v3>    # quad face (split into 2 triangles)
...
```

### Camera System

The `Camera3D` class implements:
- **Spherical coordinates**: Position defined by pitch, yaw, and distance
- **Look-at transformation**: Always looks at mesh center
- **Perspective projection**: 45° FOV with adjustable near/far planes

### Ray-Casting

Vertex picking uses OpenGL's `gluUnProject`:
1. Get near and far points in world space from screen coordinates
2. Construct a ray from camera through mouse position
3. Find closest vertex to the ray (within threshold)
4. Prioritize vertices closer to camera

### Drag Plane

When dragging a vertex:
1. Create a plane perpendicular to view direction
2. Plane passes through the selected vertex
3. Cast ray from mouse position
4. Intersect ray with plane to get new 3D position
5. Update vertex and trigger ARAP deformation

## Limitations and Future Improvements

### Current Limitations
- **2D deformation in 3D space**: Deformation is computed in 2D (XY) and Z is preserved
- **No true volumetric ARAP**: Full 3D ARAP would require tetrahedral mesh support
- **Single-resolution**: No adaptive refinement or multiresolution

### Possible Improvements
- **Full 3D ARAP**: Extend the deformation algorithm to work in true 3D
- **Tetrahedral meshes**: Support volumetric meshes
- **Multiple drag planes**: Allow choosing different drag plane orientations
- **Constraint painting**: Paint regions to pin instead of individual vertices
- **Smooth handles**: Use soft constraints with falloff
- **Animation**: Record and playback deformation sequences
- **GPU acceleration**: Move ARAP solver to GPU using compute shaders

## Compatibility

### Tested With
- Python 3.7+
- PyQt5
- PyOpenGL
- NumPy
- SciPy

### Known Issues
- Very large meshes (>10,000 vertices) may be slow to deform
- The ARAP solver uses dense matrices, which can be memory-intensive
- Some graphics drivers may have issues with multisampling (can disable in `initializeGL`)

## Credits

This extension builds upon the original 2D ARAP implementation and adds:
- 3D camera system with trackball controls
- Ray-casting for 3D vertex picking
- OFF file format support
- View-aligned drag planes for 3D manipulation
- Enhanced rendering with lighting and depth

The ARAP algorithm is based on:
*"As-Rigid-As-Possible Surface Modeling"* by Sorkine and Alexa (2007)

## License

Same as the original project.
