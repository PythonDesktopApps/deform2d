# Quick Start Guide - 3D ARAP Deformation

## Installation

```bash
pip install -r requirements.txt
```

## Running

```bash
python main.py
```

## Interface

The application has:
- **Left Panel**: Control panel with rotation mode and action buttons
- **Main View**: 3D viewport showing the mesh
- **Status Bar**: Quick help at the bottom

## Controls at a Glance

| Action | Control |
|--------|---------|
| **Pin/Unpin vertex** | Right-click on vertex |
| **Deform mesh** | Left-click vertex + drag |
| **Rotate object/camera** | Left-click empty space + drag |
| **Zoom** | Mouse wheel |
| **Load mesh** | Press `F` or click "Load Mesh" button |
| **Reset view** | Press `R` or click "Reset View" button |
| **Clear all pins** | Press `C` or click "Clear Pins" button |
| **Switch rotation mode** | Use radio buttons in left panel |

## Keyboard Shortcuts

| Key | Action | Also Available Via |
|-----|--------|-------------------|
| **F** | Load mesh | "Load Mesh" button |
| **R** | Reset view | "Reset View" button |
| **C** | Clear pins | "Clear Pins" button |

## Rotation Modes

**Rotate Object** (Default) ⭐
- The object rotates **in the direction you drag**
- Drag left → object rotates left
- Drag right → object rotates right
- Feels like spinning the mesh with your mouse
- **Most intuitive** for examining the model

**Rotate Camera**
- The camera orbits around the stationary object
- Drag left → camera moves left (object appears to rotate right)
- Drag right → camera moves right (object appears to rotate left)
- Like walking around a sculpture
- Alternative viewing mode

## Quick Tutorial

### Step 1: Load the Armadillo
The armadillo mesh should load automatically. If not:
- Press `F`
- Navigate to `assets/armadillo_250.off`
- Click Open

### Step 2: Get a Good View
- Left-click and drag in empty space to rotate
- Use mouse wheel to zoom
- Find an angle you like

### Step 3: Pin Some Vertices
- Right-click on 2-3 vertices on the armadillo
- They'll turn red (these stay fixed)
- Good spots: feet, tail, ears

### Step 4: Deform!
- Left-click on any vertex (like the nose or an ear)
- Drag it around
- Watch the mesh stretch like rubber!

### Step 5: Experiment
- Add more pins for more control
- Remove pins by right-clicking them again
- Try different mesh parts
- Press `C` to start over

## Tips

✅ **DO:**
- Use at least 2 pins (3-4 is often better)
- Pin vertices at extremities (feet, hands, head)
- Rotate the view to see deformation from different angles
- Experiment with different pin configurations

❌ **DON'T:**
- Drag too far too fast (may cause instability)
- Use just 1 pin (won't work)
- Pin vertices that are very close together

## Troubleshooting

**Q: Nothing happens when I drag?**  
A: You need to pin at least 2 vertices first (right-click on 2 different vertices)

**Q: Mesh looks weird/broken?**  
A: Press `C` to clear pins and start over

**Q: Can't see the mesh?**  
A: Press `R` to reset camera

**Q: How do I load my own mesh?**  
A: Press `F` and select a `.obj` or `.off` file

## File Formats Supported

- **.obj** - Wavefront OBJ (most common)
- **.off** - Object File Format (3D mesh format)

## Examples

Try these meshes:
- `assets/armadillo_250.off` - 3D armadillo (default)
- `assets/man.obj` - 2D character mesh
- Any other `.obj` or `.off` file you have

Enjoy deforming! 🎨
