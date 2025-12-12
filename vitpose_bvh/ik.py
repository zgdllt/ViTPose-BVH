"""Inverse kinematics utilities and rotation estimation."""

import numpy as np
from typing import Tuple


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector."""
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return v
    return v / norm


def estimate_root_rotation(left_hip: np.ndarray, right_hip: np.ndarray, 
                          neck: np.ndarray, hips: np.ndarray) -> np.ndarray:
    """Estimate root rotation from body orientation.
    
    Args:
        left_hip: Left hip position (3D)
        right_hip: Right hip position (3D)
        neck: Neck position (3D)
        hips: Hips (root) position (3D)
    
    Returns:
        Rotation matrix (3x3)
    """
    # x_body = normalize(RHip - LHip)
    x_body = normalize(right_hip - left_hip)
    
    # y_body = normalize(Neck - Hips)
    y_body = normalize(neck - hips)
    
    # z_body = normalize(cross(x_body, y_body))
    z_body = normalize(np.cross(x_body, y_body))
    
    # Re-orthogonalize: recompute x to ensure orthogonality
    x_body = normalize(np.cross(y_body, z_body))
    
    # Build rotation matrix [x, y, z] as columns
    R = np.column_stack([x_body, y_body, z_body])
    
    return R


def rotation_matrix_to_euler(R: np.ndarray, order: str = 'ZXY') -> np.ndarray:
    """Convert rotation matrix to Euler angles.
    
    Args:
        R: Rotation matrix (3x3)
        order: Rotation order (default 'ZXY' for Blender compatibility)
    
    Returns:
        Euler angles in radians (3,)
    """
    if order == 'ZXY':
        # ZXY Euler angles (Blender default for bones)
        # Rotation order: first Z, then X, then Y
        sy = R[0, 2]
        
        # Clamp to avoid numerical issues
        sy = np.clip(sy, -1.0, 1.0)
        
        if np.abs(sy) < 0.99999:
            # Non-gimbal lock case
            x = np.arcsin(sy)
            y = np.arctan2(-R[0, 1], R[0, 0])
            z = np.arctan2(-R[1, 2], R[2, 2])
        else:
            # Gimbal lock case
            x = np.arcsin(sy)
            y = 0
            z = np.arctan2(R[1, 0], R[1, 1])
        
        return np.array([z, x, y])
    
    elif order == 'XYZ':
        # XYZ Euler angles
        sy = R[0, 2]
        sy = np.clip(sy, -1.0, 1.0)
        
        if np.abs(sy) < 0.99999:
            y = np.arcsin(sy)
            x = np.arctan2(-R[1, 2], R[2, 2])
            z = np.arctan2(-R[0, 1], R[0, 0])
        else:
            y = np.arcsin(sy)
            x = np.arctan2(R[1, 0], R[1, 1])
            z = 0
        
        return np.array([x, y, z])
    
    else:
        raise ValueError(f"Unsupported rotation order: {order}")


def compute_joint_rotation(parent_pos: np.ndarray, joint_pos: np.ndarray, 
                           child_pos: np.ndarray, reference_dir: np.ndarray = None) -> np.ndarray:
    """Compute joint rotation from positions.
    
    Args:
        parent_pos: Parent joint position (3D)
        joint_pos: Current joint position (3D)
        child_pos: Child joint position (3D)
        reference_dir: Reference direction for up vector
    
    Returns:
        Rotation matrix (3x3)
    """
    # Primary axis: direction to child
    primary = normalize(child_pos - joint_pos)
    
    # Secondary axis: perpendicular to parent-joint direction
    parent_dir = normalize(joint_pos - parent_pos)
    
    # Create orthogonal basis
    if reference_dir is not None:
        up = normalize(reference_dir)
    else:
        # Use a default up vector
        up = np.array([0, 1, 0])
    
    # Ensure up is not parallel to primary
    if np.abs(np.dot(primary, up)) > 0.99:
        up = np.array([1, 0, 0])
    
    # Build orthogonal coordinate system
    right = normalize(np.cross(up, primary))
    up = normalize(np.cross(primary, right))
    
    R = np.column_stack([right, up, primary])
    
    return R


def euler_to_rotation_matrix(euler: np.ndarray, order: str = 'ZXY') -> np.ndarray:
    """Convert Euler angles to rotation matrix.
    
    Args:
        euler: Euler angles in radians (3,)
        order: Rotation order
    
    Returns:
        Rotation matrix (3x3)
    """
    if order == 'ZXY':
        z, x, y = euler
    elif order == 'XYZ':
        x, y, z = euler
    else:
        raise ValueError(f"Unsupported rotation order: {order}")
    
    # Rotation matrices for each axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(x), -np.sin(x)],
        [0, np.sin(x), np.cos(x)]
    ])
    
    Ry = np.array([
        [np.cos(y), 0, np.sin(y)],
        [0, 1, 0],
        [-np.sin(y), 0, np.cos(y)]
    ])
    
    Rz = np.array([
        [np.cos(z), -np.sin(z), 0],
        [np.sin(z), np.cos(z), 0],
        [0, 0, 1]
    ])
    
    if order == 'ZXY':
        R = Rz @ Rx @ Ry
    elif order == 'XYZ':
        R = Rx @ Ry @ Rz
    
    return R
