"""
Camera Control System
=====================
Comprehensive camera control for 3D brain visualizations.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Union


class CameraController:
    """
    Comprehensive camera control system for 3D brain visualizations.
    Provides preset views, smooth transitions, and manual control.
    """

    # Preset camera positions for standard neuroimaging views
    PRESET_VIEWS = {
        'anterior': {
            'name': 'Anterior (Front)',
            'eye': {'x': 0, 'y': 2, 'z': 0},
            'center': {'x': 0, 'y': 0, 'z': 0},
            'up': {'x': 0, 'y': 0, 'z': 1}
        },
        'posterior': {
            'name': 'Posterior (Back)',
            'eye': {'x': 0, 'y': -2, 'z': 0},
            'center': {'x': 0, 'y': 0, 'z': 0},
            'up': {'x': 0, 'y': 0, 'z': 1}
        },
        'left': {
            'name': 'Left Lateral',
            'eye': {'x': -2, 'y': 0, 'z': 0},
            'center': {'x': 0, 'y': 0, 'z': 0},
            'up': {'x': 0, 'y': 0, 'z': 1}
        },
        'right': {
            'name': 'Right Lateral',
            'eye': {'x': 2, 'y': 0, 'z': 0},
            'center': {'x': 0, 'y': 0, 'z': 0},
            'up': {'x': 0, 'y': 0, 'z': 1}
        },
        'superior': {
            'name': 'Superior (Top)',
            'eye': {'x': 0, 'y': 0, 'z': 2},
            'center': {'x': 0, 'y': 0, 'z': 0},
            'up': {'x': 0, 'y': 1, 'z': 0}
        },
        'inferior': {
            'name': 'Inferior (Bottom)',
            'eye': {'x': 0, 'y': 0, 'z': -2},
            'center': {'x': 0, 'y': 0, 'z': 0},
            'up': {'x': 0, 'y': -1, 'z': 0}
        },
        'anterolateral_left': {
            'name': 'Anterolateral Left',
            'eye': {'x': -1.5, 'y': 1.5, 'z': 0.5},
            'center': {'x': 0, 'y': 0, 'z': 0},
            'up': {'x': 0, 'y': 0, 'z': 1}
        },
        'anterolateral_right': {
            'name': 'Anterolateral Right',
            'eye': {'x': 1.5, 'y': 1.5, 'z': 0.5},
            'center': {'x': 0, 'y': 0, 'z': 0},
            'up': {'x': 0, 'y': 0, 'z': 1}
        },
        'posterolateral_left': {
            'name': 'Posterolateral Left',
            'eye': {'x': -1.5, 'y': -1.5, 'z': 0.5},
            'center': {'x': 0, 'y': 0, 'z': 0},
            'up': {'x': 0, 'y': 0, 'z': 1}
        },
        'posterolateral_right': {
            'name': 'Posterolateral Right',
            'eye': {'x': 1.5, 'y': -1.5, 'z': 0.5},
            'center': {'x': 0, 'y': 0, 'z': 0},
            'up': {'x': 0, 'y': 0, 'z': 1}
        },
        'oblique': {
            'name': 'Oblique View',
            'eye': {'x': 1.5, 'y': 1.5, 'z': 1.5},
            'center': {'x': 0, 'y': 0, 'z': 0},
            'up': {'x': 0, 'y': 0, 'z': 1}
        },
        'custom': {
            'name': 'Custom View',
            'eye': {'x': 1.5, 'y': 1.5, 'z': 1.5},
            'center': {'x': 0, 'y': 0, 'z': 0},
            'up': {'x': 0, 'y': 0, 'z': 1}
        }
    }

    @classmethod
    def get_camera_position(cls, view_name: str = 'oblique') -> Dict:
        """Get camera position for a named view."""
        if view_name in cls.PRESET_VIEWS:
            return cls.PRESET_VIEWS[view_name].copy()
        else:
            print(f"Warning: Unknown view '{view_name}', using oblique view")
            return cls.PRESET_VIEWS['oblique'].copy()

    @classmethod
    def create_camera_from_angles(cls, azimuth: float, elevation: float,
                                  distance: float = 2.0) -> Dict:
        """
        Create camera position from spherical coordinates.

        Parameters
        ----------
        azimuth : float
            Horizontal rotation angle in degrees (0-360)
        elevation : float
            Vertical rotation angle in degrees (-90 to 90)
        distance : float
            Distance from origin
        """
        # Convert to radians
        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)

        # Calculate eye position
        x = distance * np.cos(el_rad) * np.cos(az_rad)
        y = distance * np.cos(el_rad) * np.sin(az_rad)
        z = distance * np.sin(el_rad)

        return {
            'name': f'Custom (Az:{azimuth:.0f}, El:{elevation:.0f})',
            'eye': {'x': x, 'y': y, 'z': z},
            'center': {'x': 0, 'y': 0, 'z': 0},
            'up': {'x': 0, 'y': 0, 'z': 1}
        }

    @classmethod
    def create_camera_from_coordinates(cls, x: float, y: float, z: float,
                                       center_x: float = 0, center_y: float = 0,
                                       center_z: float = 0) -> Dict:
        """
        Create camera position from Cartesian coordinates.

        Parameters
        ----------
        x, y, z : float
            Eye position coordinates
        center_x, center_y, center_z : float
            Center point coordinates (what the camera looks at)
        """
        return {
            'name': f'Custom (X:{x:.1f}, Y:{y:.1f}, Z:{z:.1f})',
            'eye': {'x': x, 'y': y, 'z': z},
            'center': {'x': center_x, 'y': center_y, 'z': center_z},
            'up': {'x': 0, 'y': 0, 'z': 1}
        }

    @classmethod
    def save_camera_position(cls, camera_dict: Dict, filepath: Union[str, Path]):
        """Save camera position to JSON file."""
        filepath = Path(filepath)
        with open(filepath, 'w') as f:
            json.dump(camera_dict, f, indent=2)
        print(f"Camera position saved to: {filepath}")

    @classmethod
    def load_camera_position(cls, filepath: Union[str, Path]) -> Dict:
        """Load camera position from JSON file."""
        filepath = Path(filepath)
        if filepath.exists():
            with open(filepath, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"Camera position file not found: {filepath}")

    @classmethod
    def interpolate_cameras(cls, camera1: Dict, camera2: Dict, steps: int = 30) -> List[Dict]:
        """
        Create smooth transition between two camera positions.

        Parameters
        ----------
        camera1, camera2 : dict
            Start and end camera positions
        steps : int
            Number of interpolation steps
        """
        frames = []
        for i in range(steps + 1):
            t = i / steps
            frame = {
                'eye': {
                    'x': camera1['eye']['x'] + t * (camera2['eye']['x'] - camera1['eye']['x']),
                    'y': camera1['eye']['y'] + t * (camera2['eye']['y'] - camera1['eye']['y']),
                    'z': camera1['eye']['z'] + t * (camera2['eye']['z'] - camera1['eye']['z'])
                },
                'center': {
                    'x': camera1['center']['x'] + t * (camera2['center']['x'] - camera1['center']['x']),
                    'y': camera1['center']['y'] + t * (camera2['center']['y'] - camera1['center']['y']),
                    'z': camera1['center']['z'] + t * (camera2['center']['z'] - camera1['center']['z'])
                },
                'up': camera1['up']  # Keep up vector constant
            }
            frames.append(frame)
        return frames
