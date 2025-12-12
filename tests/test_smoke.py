#!/usr/bin/env python
"""Smoke test for the export_video_to_bvh.py CLI script."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test imports
print("Testing imports...")

try:
    from vitpose_bvh.skeleton import (
        build_skeleton_hierarchy,
        create_virtual_joints,
        map_coco17_to_skeleton,
        compute_bone_lengths,
        lock_bone_lengths,
    )
    print("✓ vitpose_bvh.skeleton imported successfully")
except Exception as e:
    print(f"✗ Failed to import vitpose_bvh.skeleton: {e}")
    sys.exit(1)

try:
    from vitpose_bvh.filters import OneEuroFilter, MultiChannelOneEuroFilter
    print("✓ vitpose_bvh.filters imported successfully")
except Exception as e:
    print(f"✗ Failed to import vitpose_bvh.filters: {e}")
    sys.exit(1)

try:
    from vitpose_bvh.ik import (
        estimate_root_rotation,
        rotation_matrix_to_euler,
        compute_joint_rotation,
    )
    print("✓ vitpose_bvh.ik imported successfully")
except Exception as e:
    print(f"✗ Failed to import vitpose_bvh.ik: {e}")
    sys.exit(1)

try:
    from vitpose_bvh.bvh_writer import BVHWriter
    print("✓ vitpose_bvh.bvh_writer imported successfully")
except Exception as e:
    print(f"✗ Failed to import vitpose_bvh.bvh_writer: {e}")
    sys.exit(1)

# Test basic functionality
print("\nTesting basic functionality...")

import numpy as np

# Test skeleton hierarchy
hierarchy = build_skeleton_hierarchy()
print(f"✓ Built skeleton hierarchy with {len(hierarchy)} joints")

# Test virtual joints
keypoints_2d = np.random.rand(17, 2) * 100
virtual_joints = create_virtual_joints(keypoints_2d)
print(f"✓ Created {len(virtual_joints)} virtual joints")

# Test BVH writer
bvh_writer = BVHWriter(hierarchy)
print(f"✓ Created BVH writer")

# Test filter
filter_1d = OneEuroFilter()
filtered = filter_1d(np.array([1.0, 2.0, 3.0]), 0.0)
print(f"✓ OneEuro filter works")

filter_3d = MultiChannelOneEuroFilter(n_channels=3)
filtered = filter_3d(np.array([1.0, 2.0, 3.0]), 0.0)
print(f"✓ MultiChannel OneEuro filter works")

# Test rotation estimation
left_hip = np.array([1.0, 0.0, 0.0])
right_hip = np.array([-1.0, 0.0, 0.0])
neck = np.array([0.0, 1.0, 0.0])
hips = np.array([0.0, 0.0, 0.0])
R = estimate_root_rotation(left_hip, right_hip, neck, hips)
print(f"✓ Root rotation estimation works")

euler = rotation_matrix_to_euler(R)
print(f"✓ Rotation matrix to Euler conversion works")

print("\n✓ All smoke tests passed!")
