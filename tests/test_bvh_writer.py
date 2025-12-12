"""Unit tests for BVH writer."""

import os
import sys
import tempfile
import unittest
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vitpose_bvh.bvh_writer import BVHWriter
from vitpose_bvh.skeleton import build_skeleton_hierarchy


class TestBVHWriter(unittest.TestCase):
    """Test BVH writer functionality."""
    
    def test_bvh_writer_init(self):
        """Test BVH writer initialization."""
        hierarchy = build_skeleton_hierarchy()
        writer = BVHWriter(hierarchy)
        
        assert writer.hierarchy == hierarchy
        assert 'Hips' in writer.joint_channels
        assert len(writer.joint_channels['Hips']) == 6  # 3 position + 3 rotation
        
        # Other joints should have only 3 rotation channels
        assert len(writer.joint_channels['Spine']) == 3
        assert len(writer.joint_channels['LeftArm']) == 3
    
    def test_set_offsets(self):
        """Test setting joint offsets."""
        hierarchy = build_skeleton_hierarchy()
        writer = BVHWriter(hierarchy)
        
        # Create sample joint positions
        joint_positions = {
            'Hips': np.array([0.0, 1.0, 0.0]),
            'Spine': np.array([0.0, 1.2, 0.0]),
            'Chest': np.array([0.0, 1.4, 0.0]),
            'Neck': np.array([0.0, 1.5, 0.0]),
            'Head': np.array([0.0, 1.7, 0.0]),
            'LeftShoulder': np.array([-0.1, 1.4, 0.0]),
            'LeftArm': np.array([-0.2, 1.4, 0.0]),
            'LeftForeArm': np.array([-0.4, 1.2, 0.0]),
            'LeftHand': np.array([-0.6, 1.0, 0.0]),
            'RightShoulder': np.array([0.1, 1.4, 0.0]),
            'RightArm': np.array([0.2, 1.4, 0.0]),
            'RightForeArm': np.array([0.4, 1.2, 0.0]),
            'RightHand': np.array([0.6, 1.0, 0.0]),
            'LeftUpLeg': np.array([-0.1, 0.9, 0.0]),
            'LeftLeg': np.array([-0.1, 0.5, 0.0]),
            'LeftFoot': np.array([-0.1, 0.0, 0.1]),
            'RightUpLeg': np.array([0.1, 0.9, 0.0]),
            'RightLeg': np.array([0.1, 0.5, 0.0]),
            'RightFoot': np.array([0.1, 0.0, 0.1]),
        }
        
        writer.set_offsets(joint_positions)
        
        # Root should have zero offset
        assert np.allclose(writer.joint_offsets['Hips'], [0, 0, 0])
        
        # Other joints should have non-zero offsets
        assert not np.allclose(writer.joint_offsets['Spine'], [0, 0, 0])
    
    def test_write_bvh_structure(self):
        """Test BVH file structure."""
        hierarchy = build_skeleton_hierarchy()
        writer = BVHWriter(hierarchy)
        
        # Create sample data
        joint_positions = {
            'Hips': np.array([0.0, 1.0, 0.0]),
            'Spine': np.array([0.0, 1.2, 0.0]),
            'Chest': np.array([0.0, 1.4, 0.0]),
            'Neck': np.array([0.0, 1.5, 0.0]),
            'Head': np.array([0.0, 1.7, 0.0]),
            'LeftShoulder': np.array([-0.1, 1.4, 0.0]),
            'LeftArm': np.array([-0.2, 1.4, 0.0]),
            'LeftForeArm': np.array([-0.4, 1.2, 0.0]),
            'LeftHand': np.array([-0.6, 1.0, 0.0]),
            'RightShoulder': np.array([0.1, 1.4, 0.0]),
            'RightArm': np.array([0.2, 1.4, 0.0]),
            'RightForeArm': np.array([0.4, 1.2, 0.0]),
            'RightHand': np.array([0.6, 1.0, 0.0]),
            'LeftUpLeg': np.array([-0.1, 0.9, 0.0]),
            'LeftLeg': np.array([-0.1, 0.5, 0.0]),
            'LeftFoot': np.array([-0.1, 0.0, 0.1]),
            'RightUpLeg': np.array([0.1, 0.9, 0.0]),
            'RightLeg': np.array([0.1, 0.5, 0.0]),
            'RightFoot': np.array([0.1, 0.0, 0.1]),
        }
        
        writer.set_offsets(joint_positions)
        
        # Create motion data (3 frames)
        motion_data = [joint_positions for _ in range(3)]
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.bvh', delete=False) as f:
            temp_path = f.name
        
        try:
            writer.write(temp_path, motion_data, frame_time=1.0/30.0)
            
            # Read and verify file
            with open(temp_path, 'r') as f:
                content = f.read()
            
            # Check structure
            assert content.startswith('HIERARCHY')
            assert 'MOTION' in content
            assert 'Frames: 3' in content
            assert 'Frame Time:' in content
            
            # Check hierarchy elements
            assert 'ROOT Hips' in content
            assert 'JOINT Spine' in content
            assert 'JOINT Chest' in content
            assert 'OFFSET' in content
            assert 'CHANNELS' in content
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_write_with_rotations(self):
        """Test writing BVH with rotation data."""
        hierarchy = build_skeleton_hierarchy()
        writer = BVHWriter(hierarchy)
        
        # Create sample data
        joint_positions = {
            'Hips': np.array([0.0, 1.0, 0.0]),
            'Spine': np.array([0.0, 1.2, 0.0]),
            'Chest': np.array([0.0, 1.4, 0.0]),
            'Neck': np.array([0.0, 1.5, 0.0]),
            'Head': np.array([0.0, 1.7, 0.0]),
            'LeftShoulder': np.array([-0.1, 1.4, 0.0]),
            'LeftArm': np.array([-0.2, 1.4, 0.0]),
            'LeftForeArm': np.array([-0.4, 1.2, 0.0]),
            'LeftHand': np.array([-0.6, 1.0, 0.0]),
            'RightShoulder': np.array([0.1, 1.4, 0.0]),
            'RightArm': np.array([0.2, 1.4, 0.0]),
            'RightForeArm': np.array([0.4, 1.2, 0.0]),
            'RightHand': np.array([0.6, 1.0, 0.0]),
            'LeftUpLeg': np.array([-0.1, 0.9, 0.0]),
            'LeftLeg': np.array([-0.1, 0.5, 0.0]),
            'LeftFoot': np.array([-0.1, 0.0, 0.1]),
            'RightUpLeg': np.array([0.1, 0.9, 0.0]),
            'RightLeg': np.array([0.1, 0.5, 0.0]),
            'RightFoot': np.array([0.1, 0.0, 0.1]),
        }
        
        writer.set_offsets(joint_positions)
        
        # Create rotation data (5 frames)
        n_frames = 5
        root_positions = np.array([[0.0, 1.0, 0.0] for _ in range(n_frames)])
        
        joint_rotations = {}
        for joint_name, _, _ in hierarchy:
            # Small rotation values in degrees
            joint_rotations[joint_name] = np.random.randn(n_frames, 3) * 5.0
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.bvh', delete=False) as f:
            temp_path = f.name
        
        try:
            writer.write_with_rotations(
                temp_path,
                root_positions,
                joint_rotations,
                frame_time=1.0/30.0
            )
            
            # Read and verify file
            with open(temp_path, 'r') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Check basic structure
            assert lines[0] == 'HIERARCHY'
            assert 'MOTION' in content
            assert f'Frames: {n_frames}' in content
            
            # Find motion section
            motion_idx = next(i for i, line in enumerate(lines) if line == 'MOTION')
            
            # Count data lines (should be n_frames)
            data_lines = [line for line in lines[motion_idx+3:] if line.strip() and not line.startswith('Frame')]
            assert len(data_lines) == n_frames
            
            # Check each frame has correct number of values
            for line in data_lines:
                values = line.split()
                # Count channels: root has 6, each other joint has 3
                num_joints = len(hierarchy)
                expected_channels = 6 + (num_joints - 1) * 3
                assert len(values) == expected_channels
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_channel_count_consistency(self):
        """Test that channel counts are consistent across hierarchy and motion."""
        hierarchy = build_skeleton_hierarchy()
        writer = BVHWriter(hierarchy)
        
        # Calculate expected channel count
        total_channels = 0
        for joint_name, _, _ in hierarchy:
            channels = writer.joint_channels.get(joint_name, [])
            total_channels += len(channels)
        
        # Root should have 6, others should have 3
        num_joints = len(hierarchy)
        expected_total = 6 + (num_joints - 1) * 3
        
        assert total_channels == expected_total


if __name__ == '__main__':
    unittest.main()
