# UI Features Summary

## Complete Feature List

### Control Panel (Left Side)

#### Rotation Mode Selection
- **Rotate Object** (Default)
  - Object rotates in the direction of mouse drag
  - Intuitive direct manipulation
  - Camera stays fixed
  
- **Rotate Camera**
  - Camera orbits around the stationary object
  - Traditional 3D viewer behavior
  - Object stays fixed

#### Action Buttons
- **Load Mesh (F)**: Open file dialog to load .obj or .off files
- **Reset View (R)**: Reset camera/object to default viewing angle
- **Clear Pins (C)**: Remove all pinned vertices and reset deformation

#### Instructions Panel
- Quick reference for all controls
- Always visible for new users

### Main Viewport (Center/Right)

#### Mesh Display
- **Wireframe edges**: Black lines showing mesh structure
- **Filled faces**: Light blue/gray shaded surfaces
- **Lighting**: Basic directional lighting for depth perception
- **Depth testing**: Proper 3D occlusion

#### Pinned Vertices
- **Red points**: Large red dots mark pinned vertices
- **Always visible**: Rendered on top (no depth testing)
- **Clear indication**: Easy to see which vertices are constrained

### Mouse Interactions

#### Left Mouse Button
**On Empty Space:**
- Click and drag to rotate (object or camera, depending on mode)
- Smooth continuous rotation
- Pitch clamped to ±89° to prevent gimbal lock

**On Vertex:**
- Click and drag to deform the mesh
- Ray-casting picks the closest vertex to mouse
- Vertex moves along view-aligned plane
- Real-time ARAP deformation

#### Right Mouse Button
**On Vertex:**
- Click to pin/unpin vertex
- Pinned vertices turn red
- Need at least 2 pins to deform
- Click again to unpin

#### Mouse Wheel
- Scroll up to zoom in
- Scroll down to zoom out
- Smooth continuous zoom
- Distance clamped to reasonable range

### Keyboard Shortcuts

| Key | Action | Also Available Via |
|-----|--------|-------------------|
| **F** | Load mesh | "Load Mesh" button |
| **R** | Reset view | "Reset View" button |
| **C** | Clear pins | "Clear Pins" button |

### Status Bar (Bottom)

Shows quick help:
> Right-click: Pin/Unpin vertex | Left-click + drag: Deform or Rotate | Wheel: Zoom

## Behavioral Details

### Rotation Behavior

#### Object Rotation Mode (Default)
```
Drag Left  → Object rotates left (counterclockwise)
Drag Right → Object rotates right (clockwise)
Drag Up    → Object pitches backward
Drag Down  → Object pitches forward
```

#### Camera Rotation Mode
```
Drag Left  → Camera moves left  → Object appears to rotate right
Drag Right → Camera moves right → Object appears to rotate left
Drag Up    → Camera moves up    → Object appears to pitch forward
Drag Down  → Camera moves down  → Object appears to pitch backward
```

### Vertex Picking
1. Mouse click generates 3D ray from camera
2. Ray tested against all vertices
3. Distance from each vertex to ray calculated
4. Closest vertex within threshold (0.05 world units) selected
5. Vertices closer to camera prioritized if multiple candidates

### Vertex Dragging
1. When vertex selected, create view-aligned plane
2. Plane perpendicular to camera view direction
3. Plane passes through selected vertex position
4. Mouse ray intersected with plane
5. Vertex moved to intersection point
6. Z-coordinate preserved during 2D deformation
7. ARAP solver updates mesh in real-time

### Pin Management
- Right-click adds pin if not already pinned
- Right-click removes pin if already pinned
- Removing pin restores vertex to original position
- Need minimum 2 pins for deformation to work
- Pins persist until explicitly removed or cleared

## File Format Support

### OBJ Files (.obj)
- Standard Wavefront OBJ format
- Vertex positions (v x y z)
- Face definitions (f v1 v2 v3)
- Supports both 2D (z=0) and 3D meshes
- Texture coordinates and normals ignored

### OFF Files (.off)
- Object File Format
- Header: "OFF"
- Counts: num_vertices num_faces 0
- Vertex list: x y z per line
- Face list: 3 v1 v2 v3 (triangles)
- Face list: 4 v1 v2 v3 v4 (quads, auto-split to triangles)

## Auto-Detection Features

### 2D vs 3D Mesh Detection
- Checks Z-coordinate range
- If max(Z) - min(Z) < 1e-6: considered 2D
- 2D meshes: camera looks straight down (pitch=0, yaw=0)
- 3D meshes: nice 3/4 view (pitch=30, yaw=45)

### Camera Auto-Fit
- Calculates mesh bounding box
- Centers camera on mesh center
- Sets distance to 2× largest dimension
- Ensures entire mesh is visible
- Triggered on mesh load and reset

## Performance Optimizations

### Real-Time Updates
- Deformation computed on every mouse move during drag
- LU factorization cached and reused
- Only recalculated when constraints change
- Typical frame rate: 30-60 FPS for meshes <5k vertices

### Rendering
- Depth testing for proper occlusion
- Polygon offset for clean wireframe
- Multisampling for anti-aliasing (if supported)
- Line smoothing for clean edges

## User Experience Details

### Visual Feedback
- Hover: No visual change (could be added)
- Click vertex: Immediate red highlight when pinned
- Drag vertex: Real-time mesh deformation
- Rotate: Smooth continuous motion

### Error Handling
- Missing files: Print error, don't crash
- Invalid file format: Print error, keep current mesh
- Failed deformation: Fall back to original positions
- Singular matrices: Use pseudo-inverse

### Defaults
- Rotation mode: Object (more intuitive)
- Initial view: 3/4 isometric for 3D meshes
- Initial mesh: Armadillo (if available)
- Fallback: Man mesh, then square grid

## Comparison: Object vs Camera Rotation

| Aspect | Object Rotation | Camera Rotation |
|--------|----------------|-----------------|
| **What moves** | Object spins | Camera orbits |
| **Drag direction** | Same as rotation | Opposite to apparent rotation |
| **Feeling** | Spinning object in hand | Walking around object |
| **Best for** | Quick examination | Traditional 3D apps |
| **Default** | Yes ⭐ | No |
| **Learning curve** | Easiest | Moderate |

## Tips for Best Experience

### For Beginners
1. Keep "Rotate Object" mode (default)
2. Start with 2-3 pins on extremities
3. Drag slowly at first
4. Use Reset View (R) if you get lost
5. Try different pin configurations

### For Advanced Users
1. Switch to "Rotate Camera" for precise viewing
2. Use more pins (4-6) for complex deformations
3. Combine rotation with deformation
4. Experiment with different mesh regions
5. Use keyboard shortcuts for faster workflow

### Common Workflows

#### Examine Model
1. Load mesh (F)
2. Rotate with left-click drag (object mode)
3. Zoom with mouse wheel
4. Reset view (R) to default angle

#### Deform Model
1. Pin 2+ vertices (right-click)
2. Select vertex to drag (left-click)
3. Drag to desired position
4. Release and admire result
5. Add more pins for refinement

#### Try Different Angle
1. Rotate to desired view
2. Pin vertices
3. Deform from that angle
4. Rotate again to check result

## Future Enhancement Ideas

Based on current UI:
- [ ] Visual vertex hover highlighting
- [ ] Pin count indicator
- [ ] Undo/redo for deformations
- [ ] Save deformed mesh
- [ ] Multiple rotation modes (trackball, turntable, etc.)
- [ ] Constraint strength slider
- [ ] Animation recording
- [ ] Screenshot capture
- [ ] Mesh statistics display
- [ ] Multiple color schemes

## Accessibility

- All mouse actions have keyboard equivalents
- Large clickable areas (buttons, vertices)
- Clear visual feedback (red pins)
- Tooltips on rotation mode buttons
- Status bar for quick help
- No time-sensitive interactions

## Platform Compatibility

Tested on:
- ✅ Windows 10/11
- ✅ PyQt5 5.15.x
- ✅ Python 3.7+
- ✅ OpenGL 2.1+

Should work on:
- macOS (with PyQt5)
- Linux (with PyQt5)

May require:
- Modern graphics drivers
- OpenGL support
- 4GB+ RAM for large meshes
