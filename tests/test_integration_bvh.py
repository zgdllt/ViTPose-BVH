#!/usr/bin/env python
"""Integration test: Create a synthetic BVH file and validate its structure."""

import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from vitpose_bvh.skeleton import build_skeleton_hierarchy
from vitpose_bvh.bvh_writer import BVHWriter

def create_synthetic_motion(n_frames=30):
    """Create synthetic motion data for testing."""
    # Define a simple T-pose skeleton
    joint_positions_t_pose = {
        'Hips': np.array([0.0, 1.0, 0.0]),
        'Spine': np.array([0.0, 1.2, 0.0]),
        'Chest': np.array([0.0, 1.4, 0.0]),
        'Neck': np.array([0.0, 1.5, 0.0]),
        'Head': np.array([0.0, 1.7, 0.0]),
        'LeftShoulder': np.array([-0.1, 1.4, 0.0]),
        'LeftArm': np.array([-0.2, 1.4, 0.0]),
        'LeftForeArm': np.array([-0.4, 1.4, 0.0]),
        'LeftHand': np.array([-0.6, 1.4, 0.0]),
        'RightShoulder': np.array([0.1, 1.4, 0.0]),
        'RightArm': np.array([0.2, 1.4, 0.0]),
        'RightForeArm': np.array([0.4, 1.4, 0.0]),
        'RightHand': np.array([0.6, 1.4, 0.0]),
        'LeftUpLeg': np.array([-0.1, 0.9, 0.0]),
        'LeftLeg': np.array([-0.1, 0.5, 0.0]),
        'LeftFoot': np.array([-0.1, 0.0, 0.1]),
        'RightUpLeg': np.array([0.1, 0.9, 0.0]),
        'RightLeg': np.array([0.1, 0.5, 0.0]),
        'RightFoot': np.array([0.1, 0.0, 0.1]),
    }
    
    # Create animated motion (simple up/down movement)
    motion_data = []
    root_positions = []
    joint_rotations = {}
    
    hierarchy = build_skeleton_hierarchy()
    
    for i in range(n_frames):
        t = i / n_frames
        # Simple sinusoidal motion
        offset_y = 0.1 * np.sin(2 * np.pi * t)
        
        frame_positions = {}
        for joint_name, pos in joint_positions_t_pose.items():
            # Add vertical offset to create motion
            frame_positions[joint_name] = pos + np.array([0.0, offset_y, 0.0])
        
        motion_data.append(frame_positions)
        root_positions.append(frame_positions['Hips'])
    
    root_positions = np.array(root_positions)
    
    # Create rotation data (zeros for simplicity)
    for joint_name, _, _ in hierarchy:
        joint_rotations[joint_name] = np.zeros((n_frames, 3))
    
    return motion_data, root_positions, joint_rotations


def validate_bvh_file(filepath):
    """Validate BVH file structure."""
    with open(filepath, 'r') as f:
        content = f.read()
        lines = content.split('\n')
    
    # Check structure
    assert content.startswith('HIERARCHY'), "BVH must start with HIERARCHY"
    assert 'MOTION' in content, "BVH must contain MOTION section"
    assert 'ROOT Hips' in content, "BVH must have ROOT Hips joint"
    assert 'Frames:' in content, "BVH must specify frame count"
    assert 'Frame Time:' in content, "BVH must specify frame time"
    
    # Check required joints
    required_joints = ['Hips', 'Spine', 'Chest', 'Neck', 'Head',
                      'LeftArm', 'RightArm', 'LeftLeg', 'RightLeg']
    for joint in required_joints:
        assert joint in content, f"BVH must contain {joint} joint"
    
    # Find motion section
    motion_idx = next(i for i, line in enumerate(lines) if line == 'MOTION')
    
    # Extract frame count
    frames_line = next(line for line in lines[motion_idx:] if line.startswith('Frames:'))
    n_frames = int(frames_line.split(':')[1].strip())
    
    # Extract frame time
    frame_time_line = next(line for line in lines[motion_idx:] if line.startswith('Frame Time:'))
    frame_time = float(frame_time_line.split(':')[1].strip())
    
    # Count data lines
    data_lines = [line for line in lines[motion_idx+3:] if line.strip() and not line.startswith('Frame')]
    actual_frames = len(data_lines)
    
    assert actual_frames == n_frames, f"Frame count mismatch: declared {n_frames}, found {actual_frames}"
    
    # Validate channel counts
    if data_lines:
        first_frame_values = data_lines[0].split()
        # Root has 6 channels (3 pos + 3 rot), other joints have 3 rot channels
        # With 19 joints: 6 + 18*3 = 60 channels
        expected_channels = 60  # This matches our hierarchy
        assert len(first_frame_values) == expected_channels, \
            f"Channel count mismatch: expected {expected_channels}, found {len(first_frame_values)}"
    
    return True


def main():
    print("Integration test: Synthetic BVH generation")
    print("=" * 50)
    
    # Create hierarchy
    hierarchy = build_skeleton_hierarchy()
    print(f"✓ Created skeleton hierarchy ({len(hierarchy)} joints)")
    
    # Create BVH writer
    writer = BVHWriter(hierarchy)
    print("✓ Created BVH writer")
    
    # Generate synthetic motion
    n_frames = 30
    fps = 30
    motion_data, root_positions, joint_rotations = create_synthetic_motion(n_frames)
    print(f"✓ Generated {n_frames} frames of synthetic motion at {fps} FPS")
    
    # Set offsets from T-pose
    writer.set_offsets(motion_data[0])
    print("✓ Set skeleton offsets from T-pose")
    
    # Write BVH file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.bvh', delete=False) as f:
        temp_path = f.name
    
    try:
        frame_time = 1.0 / fps
        writer.write_with_rotations(temp_path, root_positions, joint_rotations, frame_time)
        print(f"✓ Wrote BVH file to {temp_path}")
        
        # Validate structure
        validate_bvh_file(temp_path)
        print("✓ BVH file structure validated")
        
        # Show file info
        with open(temp_path, 'r') as f:
            lines = f.readlines()
        
        print(f"\nBVH File Summary:")
        print(f"  Total lines: {len(lines)}")
        print(f"  Frames: {n_frames}")
        print(f"  FPS: {fps}")
        print(f"  Duration: {n_frames / fps:.2f}s")
        
        # Show first few lines
        print(f"\nFirst 10 lines:")
        for i, line in enumerate(lines[:10]):
            print(f"  {i+1:2d}: {line.rstrip()}")
        
        print("\n✓ Integration test passed!")
        
        # Optional: keep the file for manual inspection
        print(f"\nGenerated BVH file: {temp_path}")
        print("(Delete manually after inspection)")
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


if __name__ == '__main__':
    main()
