"""
Mesh Loading Utilities
======================
Shared mesh loading functions for brain visualization.

Supported file formats:
- GIFTI (.gii) - Neuroimaging mesh format
- OBJ (.obj) - Wavefront OBJ format
- MZ3 (.mz3) - Surfice mesh format
- PLY (.ply) - Polygon File Format
"""

import numpy as np
import nibabel as nib
import struct
import gzip
from pathlib import Path
from typing import Tuple, Union, Optional


# Supported file extensions
SUPPORTED_EXTENSIONS = {'.gii', '.obj', '.mz3', '.ply'}


def _load_gifti(mesh_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load mesh from GIFTI (.gii) file.

    Parameters
    ----------
    mesh_path : Path
        Path to .gii mesh file

    Returns
    -------
    tuple
        (vertices, faces) as numpy arrays
    """
    print(f"  Detected GIFTI format (.gii)")

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
            raise ValueError(f"Could not extract vertices and faces from GIFTI file. Found {len(gii.darrays)} arrays.")

    return vertices, faces


def _load_obj(mesh_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load mesh from Wavefront OBJ (.obj) file.

    Parameters
    ----------
    mesh_path : Path
        Path to .obj mesh file

    Returns
    -------
    tuple
        (vertices, faces) as numpy arrays
    """
    print(f"  Detected OBJ format (.obj)")

    vertices = []
    faces = []

    with open(mesh_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if not parts:
                continue

            if parts[0] == 'v':
                # Vertex line: v x y z [w]
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                # Face line: f v1[/vt1/vn1] v2[/vt2/vn2] v3[/vt3/vn3] ...
                # We only need vertex indices, which come before any slashes
                face_vertices = []
                for vertex_data in parts[1:]:
                    # Split by '/' and take first element (vertex index)
                    vertex_idx = vertex_data.split('/')[0]
                    # OBJ indices are 1-based, convert to 0-based
                    face_vertices.append(int(vertex_idx) - 1)

                # Handle polygons with more than 3 vertices by triangulating
                if len(face_vertices) >= 3:
                    # Simple fan triangulation for convex polygons
                    for i in range(1, len(face_vertices) - 1):
                        faces.append([face_vertices[0], face_vertices[i], face_vertices[i + 1]])

    if not vertices:
        raise ValueError("No vertices found in OBJ file")
    if not faces:
        raise ValueError("No faces found in OBJ file")

    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32)

    return vertices, faces


def _load_mz3(mesh_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load mesh from Surfice MZ3 (.mz3) file.

    MZ3 is a binary format used by Surfice/MRIcroGL. Files may be gzip compressed.

    File structure:
    - 2 bytes: magic number (0x4D5A = "MZ")
    - 2 bytes: attributes (bit flags)
    - 4 bytes: number of faces
    - 4 bytes: number of vertices
    - 4 bytes: number of skip bytes
    - [skip bytes]: optional data to skip
    - [faces]: if attr & 1, int32 face indices (nFaces * 3)
    - [vertices]: float32 vertex positions (nVertices * 3)
    - [vertex colors]: if attr & 2, RGBA colors (nVertices * 4 bytes)
    - [scalars]: if attr & 4, float32 scalars (nVertices)

    Parameters
    ----------
    mesh_path : Path
        Path to .mz3 mesh file

    Returns
    -------
    tuple
        (vertices, faces) as numpy arrays
    """
    print(f"  Detected MZ3 format (.mz3)")

    # Check if file is gzip compressed by reading magic bytes
    with open(mesh_path, 'rb') as f:
        magic_check = f.read(2)

    # Gzip magic number is 0x1f 0x8b
    is_gzipped = (magic_check == b'\x1f\x8b')

    if is_gzipped:
        print(f"  MZ3 file is gzip compressed, decompressing...")
        open_func = gzip.open
    else:
        open_func = open

    with open_func(mesh_path, 'rb') as f:
        # Read header
        magic = f.read(2)
        if magic != b'MZ':
            raise ValueError(f"Invalid MZ3 file: magic number mismatch (got {magic})")

        attr = struct.unpack('<H', f.read(2))[0]  # uint16, little endian
        n_faces = struct.unpack('<I', f.read(4))[0]  # uint32
        n_vertices = struct.unpack('<I', f.read(4))[0]  # uint32
        n_skip = struct.unpack('<I', f.read(4))[0]  # uint32

        print(f"  MZ3 header: {n_vertices} vertices, {n_faces} faces, attr={attr}, skip={n_skip}")

        # Skip optional header data
        if n_skip > 0:
            f.read(n_skip)

        # Read faces (if present - attr bit 0)
        faces = None
        if attr & 1:
            # Faces are stored as flat int32 array
            face_data = np.frombuffer(f.read(n_faces * 3 * 4), dtype=np.int32)
            faces = face_data.reshape((n_faces, 3))
        else:
            raise ValueError("MZ3 file does not contain face data")

        # Read vertices
        vertex_data = np.frombuffer(f.read(n_vertices * 3 * 4), dtype=np.float32)
        vertices = vertex_data.reshape((n_vertices, 3))

        # We skip vertex colors (attr & 2) and scalars (attr & 4) as we only need geometry

    return vertices, faces


def _load_ply(mesh_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load mesh from PLY (.ply) file.

    Supports both ASCII and binary (little/big endian) PLY formats.

    Parameters
    ----------
    mesh_path : Path
        Path to .ply mesh file

    Returns
    -------
    tuple
        (vertices, faces) as numpy arrays
    """
    print(f"  Detected PLY format (.ply)")

    with open(mesh_path, 'rb') as f:
        # Parse header
        line = f.readline().decode('ascii').strip()
        if line != 'ply':
            raise ValueError("Invalid PLY file: missing 'ply' header")

        format_type = None
        n_vertices = 0
        n_faces = 0
        vertex_properties = []
        face_properties = []
        current_element = None

        while True:
            line = f.readline().decode('ascii').strip()
            if line == 'end_header':
                break

            parts = line.split()
            if not parts:
                continue

            if parts[0] == 'format':
                format_type = parts[1]  # 'ascii', 'binary_little_endian', or 'binary_big_endian'
            elif parts[0] == 'element':
                current_element = parts[1]
                if current_element == 'vertex':
                    n_vertices = int(parts[2])
                elif current_element == 'face':
                    n_faces = int(parts[2])
            elif parts[0] == 'property':
                if current_element == 'vertex':
                    vertex_properties.append(parts)
                elif current_element == 'face':
                    face_properties.append(parts)

        print(f"  PLY format: {format_type}, {n_vertices} vertices, {n_faces} faces")

        if format_type == 'ascii':
            vertices, faces = _parse_ply_ascii(f, n_vertices, n_faces, vertex_properties)
        elif format_type in ('binary_little_endian', 'binary_big_endian'):
            endian = '<' if format_type == 'binary_little_endian' else '>'
            vertices, faces = _parse_ply_binary(f, n_vertices, n_faces, vertex_properties, face_properties, endian)
        else:
            raise ValueError(f"Unknown PLY format: {format_type}")

    return vertices, faces


def _parse_ply_ascii(f, n_vertices: int, n_faces: int, vertex_properties: list) -> Tuple[np.ndarray, np.ndarray]:
    """Parse ASCII PLY data."""
    vertices = []
    for _ in range(n_vertices):
        line = f.readline().decode('ascii').strip()
        parts = line.split()
        # First 3 values are x, y, z
        vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])

    faces = []
    for _ in range(n_faces):
        line = f.readline().decode('ascii').strip()
        parts = line.split()
        # First value is vertex count, rest are indices
        n_verts = int(parts[0])
        face_indices = [int(parts[i + 1]) for i in range(n_verts)]

        # Triangulate if needed
        if n_verts >= 3:
            for i in range(1, n_verts - 1):
                faces.append([face_indices[0], face_indices[i], face_indices[i + 1]])

    return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)


def _get_ply_dtype(type_name: str) -> Tuple[str, int]:
    """Get numpy dtype and size for PLY type names."""
    type_map = {
        'char': ('i1', 1), 'int8': ('i1', 1),
        'uchar': ('u1', 1), 'uint8': ('u1', 1),
        'short': ('i2', 2), 'int16': ('i2', 2),
        'ushort': ('u2', 2), 'uint16': ('u2', 2),
        'int': ('i4', 4), 'int32': ('i4', 4),
        'uint': ('u4', 4), 'uint32': ('u4', 4),
        'float': ('f4', 4), 'float32': ('f4', 4),
        'double': ('f8', 8), 'float64': ('f8', 8),
    }
    return type_map.get(type_name, ('f4', 4))


def _parse_ply_binary(f, n_vertices: int, n_faces: int, vertex_properties: list,
                      face_properties: list, endian: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse binary PLY data."""
    # Calculate vertex struct
    vertex_format = []
    for prop in vertex_properties:
        dtype, size = _get_ply_dtype(prop[1])
        vertex_format.append((prop[2], endian + dtype))

    vertex_dtype = np.dtype(vertex_format)
    vertex_data = np.frombuffer(f.read(n_vertices * vertex_dtype.itemsize), dtype=vertex_dtype)

    vertices = np.column_stack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).astype(np.float32)

    # Parse faces - format is typically: list uchar int vertex_indices
    faces = []
    count_dtype, count_size = _get_ply_dtype(face_properties[0][2])  # list count type
    index_dtype, index_size = _get_ply_dtype(face_properties[0][3])  # index type

    for _ in range(n_faces):
        count = np.frombuffer(f.read(count_size), dtype=endian + count_dtype)[0]
        indices = np.frombuffer(f.read(count * index_size), dtype=endian + index_dtype)

        # Triangulate if needed
        if count >= 3:
            for i in range(1, count - 1):
                faces.append([indices[0], indices[i], indices[i + 1]])

    return vertices, np.array(faces, dtype=np.int32)


def load_mesh_file(mesh_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load brain mesh from various file formats.

    Automatically detects file format based on extension and loads accordingly.

    Parameters
    ----------
    mesh_path : str or Path
        Path to mesh file. Supported formats:
        - .gii (GIFTI)
        - .obj (Wavefront OBJ)
        - .mz3 (Surfice MZ3)
        - .ply (Polygon File Format)

    Returns
    -------
    tuple
        (vertices, faces) as numpy arrays
        - vertices: array of shape (n_vertices, 3)
        - faces: array of shape (n_faces, 3)

    Raises
    ------
    FileNotFoundError
        If the mesh file does not exist
    ValueError
        If the file format is not supported or cannot be parsed
    """
    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    print(f"Loading mesh from: {mesh_path}")

    # Get file extension (lowercase)
    extension = mesh_path.suffix.lower()

    # Check if format is supported
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported mesh file format: '{extension}'\n"
            f"Supported formats are: {', '.join(sorted(SUPPORTED_EXTENSIONS))}\n"
            f"Could not detect a valid mesh format for: {mesh_path}"
        )

    # Load based on format
    if extension == '.gii':
        vertices, faces = _load_gifti(mesh_path)
    elif extension == '.obj':
        vertices, faces = _load_obj(mesh_path)
    elif extension == '.mz3':
        vertices, faces = _load_mz3(mesh_path)
    elif extension == '.ply':
        vertices, faces = _load_ply(mesh_path)
    else:
        # This shouldn't happen due to the check above, but just in case
        raise ValueError(f"Could not detect a valid mesh format for: {mesh_path}")

    # Validate output
    if vertices is None or faces is None:
        raise ValueError(f"Failed to extract vertices and faces from: {mesh_path}")

    if len(vertices.shape) != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Invalid vertices shape: {vertices.shape}. Expected (n, 3)")

    if len(faces.shape) != 2 or faces.shape[1] != 3:
        raise ValueError(f"Invalid faces shape: {faces.shape}. Expected (n, 3)")

    print(f"  Successfully loaded mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")
    return vertices, faces
