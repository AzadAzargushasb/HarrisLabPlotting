"""
Mesh Loading Utilities
======================
Shared mesh loading functions for brain visualization.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Tuple, Union


def load_mesh_file(mesh_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load brain mesh from GIFTI file with proper intent checking.

    Parameters
    ----------
    mesh_path : str or Path
        Path to .gii mesh file

    Returns
    -------
    tuple
        (vertices, faces) as numpy arrays
        - vertices: array of shape (n_vertices, 3)
        - faces: array of shape (n_faces, 3)
    """
    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    print(f"Loading mesh from: {mesh_path}")

    # Load the GIFTI file
    gii = nib.load(str(mesh_path))
    vertices = None
    faces = None

    # Properly extract vertices and faces using intent codes
    for array in gii.darrays:
        if array.intent == nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET']:
            vertices = array.data
            print(f"  Found vertices array with shape: {vertices.shape}")
        elif array.intent == nib.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE']:
            faces = array.data
            print(f"  Found faces array with shape: {faces.shape}")

    # Fallback if intent codes aren't set properly
    if vertices is None or faces is None:
        print("  Warning: Could not find arrays by intent, using index-based loading")
        if len(gii.darrays) >= 2:
            vertices = gii.darrays[0].data
            faces = gii.darrays[1].data
        else:
            raise ValueError(f"Could not extract vertices and faces from mesh file. Found {len(gii.darrays)} arrays.")

    print(f"  Successfully loaded mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")
    return vertices, faces
