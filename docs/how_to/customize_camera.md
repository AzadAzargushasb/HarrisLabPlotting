# Customize the camera view

`hlplot` ships with nine preset camera views and supports arbitrary custom
angles. Pick a preset for standard orientations; build a custom dict when
you need a non-standard view for a paper figure.

## Use a preset

```bash
hlplot plot ... --camera left
```

Available presets:

| View | Description |
| --- | --- |
| `anterior` | From the front (looking at the face) |
| `posterior` | From the back |
| `left` / `right` | Sagittal views |
| `superior` | Top-down |
| `inferior` | Bottom-up |
| `anterolateral_left` / `anterolateral_right` | 3/4 front views |
| `posterolateral_left` / `posterolateral_right` | 3/4 back views |
| `oblique` | Default cinematic angle |

## Add a camera dropdown

Render the figure with all nine presets baked into a Plotly dropdown menu so
the reader can switch views in the HTML:

```bash
hlplot plot ... --enable-camera-controls
```

This is gold for sharing interactive figures with collaborators who want to
explore the network from multiple angles.

> **Don't confuse with `--show-camera-readout`.** `--enable-camera-controls`
> (default: **on**) toggles a static **dropdown of preset views** rendered in
> the HTML. `--show-camera-readout` (default: **off**) is a separate feature
> that injects a small **live JavaScript overlay** showing the current
> `eye/center/up` coordinates as you rotate the brain, plus a copy-pastable
> `--custom-camera-*` block. The two are independent: typing
> `--enable-camera-controls` is a no-op (already on) — to remove the
> dropdown use `--no-camera-controls`; to add the live readout overlay use
> `--show-camera-readout`. The readout overlay never appears in static
> PNG/SVG/PDF exports because kaleido does not run JavaScript.

## Zoom

Pass `--zoom 1.5` (or `zoom=1.5` from Python) to bring the camera 50% closer
without picking a custom view. Values above `1.0` make the brain look bigger;
below `1.0` push it further away. The flag applies to both the HTML and any
static export, and to every panel of a `--multi-view` stitched output. (This
replaces the old `--multi-view-zoom` flag, which has been removed.)

## Custom angles in Python

For non-standard angles, use the `CameraController`:

```python
from HarrisLabPlotting import CameraController

# By spherical coordinates (degrees)
cam = CameraController.create_camera_from_angles(
    azimuth=45, elevation=30, distance=2.0
)

# By Cartesian eye position
cam = CameraController.create_camera_from_coordinates(
    x=1.5, y=0.0, z=1.0,
    center_x=0, center_y=0, center_z=0,
)

fig.update_layout(scene_camera=cam)
```

## Save and reuse a camera

Once you've found a great angle interactively, save it for later use:

```python
CameraController.save_camera_position(cam, "my_paper_figure.json")

# next session
cam = CameraController.load_camera_position("my_paper_figure.json")
```

## Smooth camera animations

Interpolate between two camera positions (useful for animations or
multi-panel figures with subtle rotation):

```python
frames = CameraController.interpolate_cameras(
    camera1=CameraController.get_camera_position("left"),
    camera2=CameraController.get_camera_position("anterolateral_left"),
    steps=24,
)
```

## See also

- [Multi-view stitched export](../tutorials/legends_and_multiview.md) for
  rendering multiple camera angles into a single PNG strip.
