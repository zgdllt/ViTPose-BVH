"""Skeleton hierarchy and COCO17 keypoint mapping."""

import numpy as np
from typing import Dict, List, Tuple

# COCO17 keypoint indices
COCO17_KEYPOINTS = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16,
}


def create_virtual_joints(keypoints_2d: np.ndarray) -> Dict[str, np.ndarray]:
    """Create virtual joints from COCO17 keypoints.
    
    Args:
        keypoints_2d: COCO17 keypoints, shape (17, 2) or (17, 3)
    
    Returns:
        Dictionary of virtual joint positions
    """
    kp = COCO17_KEYPOINTS
    
    # Hips = (LHip + RHip) / 2
    hips = (keypoints_2d[kp['left_hip']] + keypoints_2d[kp['right_hip']]) / 2
    
    # Neck = (LShoulder + RShoulder) / 2
    neck = (keypoints_2d[kp['left_shoulder']] + keypoints_2d[kp['right_shoulder']]) / 2
    
    # Spine = lerp(Hips, Neck, 0.3)
    spine = hips + 0.3 * (neck - hips)
    
    # Chest = lerp(Hips, Neck, 0.7)
    chest = hips + 0.7 * (neck - hips)
    
    # Head derived from Nose
    head = keypoints_2d[kp['nose']]
    
    virtual_joints = {
        'hips': hips,
        'spine': spine,
        'chest': chest,
        'neck': neck,
        'head': head,
    }
    
    return virtual_joints


def build_skeleton_hierarchy() -> List[Tuple[str, str, List[str]]]:
    """Build Blender-friendly humanoid skeleton hierarchy.
    
    Returns:
        List of (joint_name, parent_name, children_names) tuples
    """
    hierarchy = [
        # (joint_name, parent_name, children)
        ('Hips', None, ['Spine', 'LeftUpLeg', 'RightUpLeg']),
        ('Spine', 'Hips', ['Chest']),
        ('Chest', 'Spine', ['Neck', 'LeftShoulder', 'RightShoulder']),
        ('Neck', 'Chest', ['Head']),
        ('Head', 'Neck', []),
        
        # Left arm
        ('LeftShoulder', 'Chest', ['LeftArm']),
        ('LeftArm', 'LeftShoulder', ['LeftForeArm']),
        ('LeftForeArm', 'LeftArm', ['LeftHand']),
        ('LeftHand', 'LeftForeArm', []),
        
        # Right arm
        ('RightShoulder', 'Chest', ['RightArm']),
        ('RightArm', 'RightShoulder', ['RightForeArm']),
        ('RightForeArm', 'RightArm', ['RightHand']),
        ('RightHand', 'RightForeArm', []),
        
        # Left leg
        ('LeftUpLeg', 'Hips', ['LeftLeg']),
        ('LeftLeg', 'LeftUpLeg', ['LeftFoot']),
        ('LeftFoot', 'LeftLeg', []),
        
        # Right leg
        ('RightUpLeg', 'Hips', ['RightLeg']),
        ('RightLeg', 'RightUpLeg', ['RightFoot']),
        ('RightFoot', 'RightLeg', []),
    ]
    
    return hierarchy


def map_coco17_to_skeleton(keypoints_2d: np.ndarray, virtual_joints: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Map COCO17 keypoints to skeleton joint positions.
    
    Args:
        keypoints_2d: COCO17 keypoints, shape (17, 2) or (17, 3)
        virtual_joints: Dictionary of virtual joint positions
    
    Returns:
        Dictionary mapping skeleton joint names to positions
    """
    kp = COCO17_KEYPOINTS
    
    joint_positions = {
        'Hips': virtual_joints['hips'],
        'Spine': virtual_joints['spine'],
        'Chest': virtual_joints['chest'],
        'Neck': virtual_joints['neck'],
        'Head': virtual_joints['head'],
        
        # Left arm
        'LeftShoulder': keypoints_2d[kp['left_shoulder']],
        'LeftArm': keypoints_2d[kp['left_shoulder']],  # LeftShoulder position
        'LeftForeArm': keypoints_2d[kp['left_elbow']],
        'LeftHand': keypoints_2d[kp['left_wrist']],
        
        # Right arm
        'RightShoulder': keypoints_2d[kp['right_shoulder']],
        'RightArm': keypoints_2d[kp['right_shoulder']],  # RightShoulder position
        'RightForeArm': keypoints_2d[kp['right_elbow']],
        'RightHand': keypoints_2d[kp['right_wrist']],
        
        # Left leg
        'LeftUpLeg': keypoints_2d[kp['left_hip']],
        'LeftLeg': keypoints_2d[kp['left_knee']],
        'LeftFoot': keypoints_2d[kp['left_ankle']],
        
        # Right leg
        'RightUpLeg': keypoints_2d[kp['right_hip']],
        'RightLeg': keypoints_2d[kp['right_knee']],
        'RightFoot': keypoints_2d[kp['right_ankle']],
    }
    
    return joint_positions


def compute_bone_lengths(joint_positions: Dict[str, np.ndarray], hierarchy: List[Tuple[str, str, List[str]]]) -> Dict[str, float]:
    """Compute bone lengths from joint positions.
    
    Args:
        joint_positions: Dictionary mapping joint names to positions
        hierarchy: Skeleton hierarchy
    
    Returns:
        Dictionary mapping bone names (child joint) to lengths
    """
    bone_lengths = {}
    
    for joint_name, parent_name, _ in hierarchy:
        if parent_name is not None and joint_name in joint_positions and parent_name in joint_positions:
            length = np.linalg.norm(joint_positions[joint_name] - joint_positions[parent_name])
            bone_lengths[joint_name] = length
    
    return bone_lengths


def lock_bone_lengths(joint_positions: Dict[str, np.ndarray], 
                     target_bone_lengths: Dict[str, float],
                     hierarchy: List[Tuple[str, str, List[str]]]) -> Dict[str, np.ndarray]:
    """Lock bone lengths to target values by projecting joints.
    
    Args:
        joint_positions: Dictionary mapping joint names to positions
        target_bone_lengths: Dictionary mapping bone names to target lengths
        hierarchy: Skeleton hierarchy
    
    Returns:
        Dictionary of adjusted joint positions
    """
    adjusted_positions = {}
    
    # Start with root (Hips)
    for joint_name, parent_name, children in hierarchy:
        if parent_name is None:
            # Root joint stays in place
            adjusted_positions[joint_name] = joint_positions[joint_name].copy()
        elif parent_name in adjusted_positions and joint_name in target_bone_lengths:
            # Project to fixed bone length
            parent_pos = adjusted_positions[parent_name]
            current_pos = joint_positions[joint_name]
            direction = current_pos - parent_pos
            direction_norm = np.linalg.norm(direction)
            
            if direction_norm > 1e-6:
                direction = direction / direction_norm
                adjusted_positions[joint_name] = parent_pos + direction * target_bone_lengths[joint_name]
            else:
                # Fallback if direction is degenerate
                adjusted_positions[joint_name] = current_pos.copy()
        else:
            adjusted_positions[joint_name] = joint_positions[joint_name].copy()
    
    return adjusted_positions
