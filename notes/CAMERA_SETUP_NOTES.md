# Camera Setup Notes

## Default Camera Angles for 3D Meshes

The camera uses spherical coordinates:
- **Pitch**: Rotation around the horizontal axis (up/down) - Range: -89° to 89°
- **Yaw**: Rotation around the vertical axis (left/right) - Range: 0° to 360°
- **Distance**: How far from the center point

## Current Default for 3D Meshes
```python
pitch = 20.0   # Looking slightly down from above
yaw = 180.0    # Facing forward (depends on mesh orientation)
```

## Common View Angles

### Front Views
- `yaw = 180.0` - Front (current default)
- `yaw = 0.0` - Back
- `yaw = 90.0` - Right side
- `yaw = 270.0` - Left side

### Isometric/3/4 Views
- `pitch = 30.0, yaw = 45.0` - Front-right isometric
- `pitch = 30.0, yaw = 135.0` - Front-left isometric
- `pitch = 30.0, yaw = 225.0` - Back-right isometric
- `pitch = 30.0, yaw = 315.0` - Back-left isometric

### Top/Bottom Views
- `pitch = 89.0, yaw = 0.0` - Top-down view
- `pitch = -89.0, yaw = 0.0` - Bottom-up view

## Adjusting for Specific Meshes

If the armadillo appears backwards, try these in `fit_camera_to_mesh()`:

```python
# Option 1: Try different yaw angles
self.camera.rotation = np.array([20.0, 0.0], dtype=np.float32)

# Option 2: Nice isometric view
self.camera.rotation = np.array([25.0, 45.0], dtype=np.float32)

# Option 3: Higher angle view
self.camera.rotation = np.array([35.0, 135.0], dtype=np.float32)
```

## Controls Reminder
- **Press R** to reset to default view
- **Left-click + drag** to manually rotate to any angle you like
- **Mouse wheel** to zoom in/out

The user can always rotate to find the best view manually!
