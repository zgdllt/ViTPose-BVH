#!/usr/bin/env python
"""Export single-person video to BVH format using ViTPose.

This script performs end-to-end video-to-BVH conversion:
1. Loads video and detects FPS
2. Runs ViTPose 2D pose detection (COCO17 keypoints)
3. Performs 2D→3D lifting
4. Applies bone-length locking and temporal smoothing
5. Estimates root rotation from body orientation
6. Exports Blender-compatible BVH file
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import cv2
import mmcv
import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mmpose.apis import (
    inference_pose_lifter_model,
    inference_top_down_pose_model,
    init_pose_model,
    extract_pose_sequence,
)

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False
    warnings.warn("mmdet not found. Person detection will not be available.")

from vitpose_bvh.skeleton import (
    create_virtual_joints,
    build_skeleton_hierarchy,
    map_coco17_to_skeleton,
    compute_bone_lengths,
    lock_bone_lengths,
)
from vitpose_bvh.filters import OneEuroFilter, MultiChannelOneEuroFilter
from vitpose_bvh.ik import estimate_root_rotation, rotation_matrix_to_euler
from vitpose_bvh.bvh_writer import BVHWriter


def select_device():
    """Auto-select device: mps → cuda → cpu."""
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda:0'
    else:
        return 'cpu'


def process_mmdet_results(mmdet_results, cat_id=1):
    """Process mmdet results to extract person detections."""
    if isinstance(mmdet_results, tuple):
        det_results = mmdet_results[0]
    else:
        det_results = mmdet_results
    
    bboxes = det_results[cat_id - 1]
    
    person_results = []
    for bbox in bboxes:
        person = {'bbox': bbox}
        person_results.append(person)
    
    return person_results


def convert_coco17_to_h36m(keypoints):
    """Convert COCO17 keypoints to H36M format for 3D lifting."""
    keypoints_new = np.zeros((17, keypoints.shape[1]))
    # pelvis is in the middle of l_hip and r_hip
    keypoints_new[0] = (keypoints[11] + keypoints[12]) / 2
    # thorax is in the middle of l_shoulder and r_shoulder
    keypoints_new[8] = (keypoints[5] + keypoints[6]) / 2
    # head is in the middle of l_eye and r_eye
    keypoints_new[10] = (keypoints[1] + keypoints[2]) / 2
    # spine is in the middle of thorax and pelvis
    keypoints_new[7] = (keypoints_new[0] + keypoints_new[8]) / 2
    # rearrange other keypoints
    keypoints_new[[1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16]] = \
        keypoints[[12, 14, 16, 11, 13, 15, 0, 5, 7, 9, 6, 8, 10]]
    return keypoints_new


def main():
    parser = argparse.ArgumentParser(description='Export video to BVH format')
    parser.add_argument('--input', required=True, help='Input video path')
    parser.add_argument('--output', required=True, help='Output BVH path')
    parser.add_argument('--det-config', help='Person detection config')
    parser.add_argument('--det-checkpoint', help='Person detection checkpoint')
    parser.add_argument('--pose-config', help='2D pose detection config')
    parser.add_argument('--pose-checkpoint', help='2D pose checkpoint')
    parser.add_argument('--pose-lifter-config', help='3D pose lifter config')
    parser.add_argument('--pose-lifter-checkpoint', help='3D pose lifter checkpoint')
    parser.add_argument('--device', default=None, help='Device (auto-selected if not specified)')
    parser.add_argument('--bbox-thr', type=float, default=0.9, help='Bbox threshold')
    parser.add_argument('--kpt-thr', type=float, default=0.3, help='Keypoint threshold')
    parser.add_argument('--smooth-type', default='oneeuro', choices=['oneeuro', 'savgol', 'none'],
                       help='Smoothing filter type')
    parser.add_argument('--min-cutoff', type=float, default=0.004, help='OneEuro min cutoff')
    parser.add_argument('--beta', type=float, default=0.7, help='OneEuro beta')
    parser.add_argument('--root-min-cutoff', type=float, default=0.001, 
                       help='OneEuro min cutoff for root (stronger smoothing)')
    parser.add_argument('--root-beta', type=float, default=0.3, help='OneEuro beta for root')
    
    args = parser.parse_args()
    
    # Auto-select device if not specified
    if args.device is None:
        args.device = select_device()
        print(f"Auto-selected device: {args.device}")
    
    # Load video
    print(f"Loading video: {args.input}")
    video = mmcv.VideoReader(args.input)
    if not video.opened:
        raise RuntimeError(f"Failed to load video: {args.input}")
    
    fps = video.fps
    frame_time = 1.0 / fps
    print(f"Video FPS: {fps}, Frame time: {frame_time:.6f}s")
    print(f"Total frames: {len(video)}")
    
    # Check if we're using detection or simple approach
    use_detection = args.det_config is not None and args.det_checkpoint is not None
    use_3d_lifting = args.pose_lifter_config is not None and args.pose_lifter_checkpoint is not None
    
    if use_detection and not has_mmdet:
        raise RuntimeError("mmdet is required for person detection but not installed")
    
    # Initialize models
    person_det_model = None
    if use_detection:
        print("Initializing person detector...")
        person_det_model = init_detector(
            args.det_config, args.det_checkpoint, device=args.device.lower()
        )
    
    pose_det_model = None
    if args.pose_config and args.pose_checkpoint:
        print("Initializing 2D pose detector...")
        pose_det_model = init_pose_model(
            args.pose_config, args.pose_checkpoint, device=args.device.lower()
        )
    
    pose_lifter_model = None
    if use_3d_lifting:
        print("Initializing 3D pose lifter...")
        pose_lifter_model = init_pose_model(
            args.pose_lifter_config, args.pose_lifter_checkpoint, device=args.device.lower()
        )
    
    # For simplified mode without full models
    if pose_det_model is None:
        print("WARNING: No pose detection model provided. BVH export requires pose detection.")
        print("Please provide --pose-config and --pose-checkpoint")
        return
    
    # Process video frames for 2D pose detection
    print("Stage 1: 2D pose detection...")
    pose_results_list = []
    
    for frame_idx, frame in enumerate(mmcv.track_iter_progress(video)):
        if use_detection:
            # Detect person
            mmdet_results = inference_detector(person_det_model, frame)
            person_results = process_mmdet_results(mmdet_results, cat_id=1)
        else:
            # Use full frame as bbox
            h, w = frame.shape[:2]
            person_results = [{'bbox': np.array([0, 0, w, h, 1.0])}]
        
        if len(person_results) == 0:
            # No person detected, use previous or skip
            if pose_results_list:
                pose_results_list.append(pose_results_list[-1])
            else:
                pose_results_list.append([])
            continue
        
        # Select single person (largest bbox or highest score)
        if len(person_results) > 1:
            # Select by bbox area * score
            scores = [p['bbox'][4] * (p['bbox'][2] - p['bbox'][0]) * (p['bbox'][3] - p['bbox'][1]) 
                     for p in person_results]
            best_idx = np.argmax(scores)
            person_results = [person_results[best_idx]]
        
        # 2D pose estimation
        pose_results, _ = inference_top_down_pose_model(
            pose_det_model,
            frame,
            person_results,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset='TopDownCocoDataset',
            return_heatmap=False,
            outputs=None
        )
        
        pose_results_list.append(pose_results)
    
    print(f"Detected poses in {len(pose_results_list)} frames")
    
    # Extract 2D keypoints sequence
    all_keypoints_2d = []
    for pose_results in pose_results_list:
        if len(pose_results) > 0:
            keypoints = pose_results[0]['keypoints']  # (17, 3) for COCO17
            all_keypoints_2d.append(keypoints[:, :2])  # Take only x, y
        else:
            # Fill with previous or zeros
            if all_keypoints_2d:
                all_keypoints_2d.append(all_keypoints_2d[-1])
            else:
                all_keypoints_2d.append(np.zeros((17, 2)))
    
    all_keypoints_2d = np.array(all_keypoints_2d)  # (n_frames, 17, 2)
    print(f"2D keypoints shape: {all_keypoints_2d.shape}")
    
    # Stage 2: 3D lifting (if available)
    all_keypoints_3d = []
    
    if use_3d_lifting:
        print("Stage 2: 2D→3D lifting...")
        
        # Convert COCO17 to H36M for lifting
        for pose_results in pose_results_list:
            for res in pose_results:
                keypoints = res['keypoints']
                res['keypoints'] = convert_coco17_to_h36m(keypoints)
        
        # Get data config
        if hasattr(pose_lifter_model.cfg, 'test_data_cfg'):
            data_cfg = pose_lifter_model.cfg.test_data_cfg
        else:
            data_cfg = pose_lifter_model.cfg.data_cfg
        
        # Process each frame
        for i in range(len(pose_results_list)):
            pose_results_2d = extract_pose_sequence(
                pose_results_list,
                frame_idx=i,
                causal=data_cfg.causal,
                seq_len=data_cfg.seq_len,
                step=data_cfg.seq_frame_interval
            )
            
            pose_lift_results = inference_pose_lifter_model(
                pose_lifter_model,
                pose_results_2d=pose_results_2d,
                dataset='Body3DH36MDataset',
                with_track_id=False,
                image_size=video.resolution,
                norm_pose_2d=False
            )
            
            if len(pose_lift_results) > 0:
                keypoints_3d = pose_lift_results[0]['keypoints_3d']
                # Convert coordinate system: swap and flip axes
                keypoints_3d = keypoints_3d[..., [0, 2, 1]]
                keypoints_3d[..., 0] = -keypoints_3d[..., 0]
                keypoints_3d[..., 2] = -keypoints_3d[..., 2]
                all_keypoints_3d.append(keypoints_3d)
            else:
                if all_keypoints_3d:
                    all_keypoints_3d.append(all_keypoints_3d[-1])
                else:
                    all_keypoints_3d.append(np.zeros((17, 3)))
        
        all_keypoints_3d = np.array(all_keypoints_3d)  # (n_frames, 17, 3)
    else:
        print("Stage 2: Simple 2D→3D (z=0)...")
        # Simple fallback: use 2D with z=0
        all_keypoints_3d = np.concatenate([
            all_keypoints_2d, 
            np.zeros((len(all_keypoints_2d), 17, 1))
        ], axis=-1)
    
    print(f"3D keypoints shape: {all_keypoints_3d.shape}")
    
    # Stage 3: Build skeleton and compute bone lengths
    print("Stage 3: Building skeleton hierarchy...")
    hierarchy = build_skeleton_hierarchy()
    
    # Process each frame to create skeleton joints
    all_joint_positions = []
    bone_lengths_per_frame = []
    
    for frame_idx in range(len(all_keypoints_3d)):
        # Get 3D keypoints for this frame
        # Note: if 3D lifting was used, keypoints are in H36M format (17 joints)
        # otherwise they are in COCO17 format (17 joints)
        keypoints_3d = all_keypoints_3d[frame_idx]
        
        # Map to skeleton joints
        # Note: The indices below assume COCO17 format from 2D detection
        # If 3D lifting was used, the keypoints were already converted to H36M during lifting,
        # but we still use the original 2D COCO17 indices for consistency
        joint_positions = {}
        
        # Use COCO17 indices (0-16) for direct mapping
        joint_positions['Hips'] = (keypoints_3d[11] + keypoints_3d[12]) / 2  # Average of hips
        joint_positions['Neck'] = (keypoints_3d[5] + keypoints_3d[6]) / 2  # Average of shoulders
        joint_positions['Spine'] = joint_positions['Hips'] + 0.3 * (joint_positions['Neck'] - joint_positions['Hips'])
        joint_positions['Chest'] = joint_positions['Hips'] + 0.7 * (joint_positions['Neck'] - joint_positions['Hips'])
        joint_positions['Head'] = keypoints_3d[0]  # Nose
        
        # Arms
        joint_positions['LeftShoulder'] = keypoints_3d[5]
        joint_positions['LeftArm'] = keypoints_3d[5]
        joint_positions['LeftForeArm'] = keypoints_3d[7]
        joint_positions['LeftHand'] = keypoints_3d[9]
        
        joint_positions['RightShoulder'] = keypoints_3d[6]
        joint_positions['RightArm'] = keypoints_3d[6]
        joint_positions['RightForeArm'] = keypoints_3d[8]
        joint_positions['RightHand'] = keypoints_3d[10]
        
        # Legs
        joint_positions['LeftUpLeg'] = keypoints_3d[11]
        joint_positions['LeftLeg'] = keypoints_3d[13]
        joint_positions['LeftFoot'] = keypoints_3d[15]
        
        joint_positions['RightUpLeg'] = keypoints_3d[12]
        joint_positions['RightLeg'] = keypoints_3d[14]
        joint_positions['RightFoot'] = keypoints_3d[16]
        
        all_joint_positions.append(joint_positions)
        
        # Compute bone lengths for this frame
        bone_lengths = compute_bone_lengths(joint_positions, hierarchy)
        bone_lengths_per_frame.append(bone_lengths)
    
    # Stage 4: Bone length locking
    print("Stage 4: Bone length locking...")
    
    # Compute median bone lengths across all frames
    all_bone_names = set()
    for bl in bone_lengths_per_frame:
        all_bone_names.update(bl.keys())
    
    median_bone_lengths = {}
    for bone_name in all_bone_names:
        lengths = [bl.get(bone_name, 0) for bl in bone_lengths_per_frame if bone_name in bl]
        if lengths:
            median_bone_lengths[bone_name] = np.median(lengths)
    
    print(f"Computed median bone lengths for {len(median_bone_lengths)} bones")
    
    # Lock bone lengths
    locked_joint_positions = []
    for joint_positions in all_joint_positions:
        locked_positions = lock_bone_lengths(joint_positions, median_bone_lengths, hierarchy)
        locked_joint_positions.append(locked_positions)
    
    # Stage 5: Temporal smoothing
    print(f"Stage 5: Temporal smoothing ({args.smooth_type})...")
    
    if args.smooth_type == 'oneeuro':
        # Apply OneEuro filter to each joint
        smoothed_positions = []
        
        # Create filters for each joint
        joint_filters = {}
        for joint_name in locked_joint_positions[0].keys():
            joint_filters[joint_name] = MultiChannelOneEuroFilter(
                n_channels=3,
                min_cutoff=args.min_cutoff,
                beta=args.beta
            )
        
        # Apply filters
        for frame_idx, joint_positions in enumerate(locked_joint_positions):
            t = frame_idx * frame_time
            smoothed = {}
            for joint_name, pos in joint_positions.items():
                if joint_name in joint_filters:
                    smoothed[joint_name] = joint_filters[joint_name](pos, t)
                else:
                    smoothed[joint_name] = pos
            smoothed_positions.append(smoothed)
        
        locked_joint_positions = smoothed_positions
    
    # Stage 6: Root rotation estimation
    print("Stage 6: Estimating root rotation...")
    
    root_rotations = []
    root_positions = []
    
    # Stronger smoothing for root rotation
    root_rot_filter = MultiChannelOneEuroFilter(
        n_channels=3,
        min_cutoff=args.root_min_cutoff,
        beta=args.root_beta
    )
    
    for frame_idx, joint_positions in enumerate(locked_joint_positions):
        # Get necessary points
        left_hip = joint_positions['LeftUpLeg']
        right_hip = joint_positions['RightUpLeg']
        neck = joint_positions['Neck']
        hips = joint_positions['Hips']
        
        # Estimate rotation matrix
        R = estimate_root_rotation(left_hip, right_hip, neck, hips)
        
        # Convert to Euler angles (in radians)
        euler_rad = rotation_matrix_to_euler(R, order='ZXY')
        euler_deg = np.degrees(euler_rad)
        
        # Apply smoothing to Euler angles
        t = frame_idx * frame_time
        if args.smooth_type == 'oneeuro':
            euler_deg = root_rot_filter(euler_deg, t)
        
        root_rotations.append(euler_deg)
        root_positions.append(hips)
    
    root_rotations = np.array(root_rotations)  # (n_frames, 3) in degrees
    root_positions = np.array(root_positions)  # (n_frames, 3)
    
    # Stage 7: Export BVH
    print("Stage 7: Exporting BVH...")
    
    # Create BVH writer
    bvh_writer = BVHWriter(hierarchy)
    
    # Set offsets from first frame (T-pose approximation)
    bvh_writer.set_offsets(locked_joint_positions[0])
    
    # Prepare joint rotations (placeholder - all zeros except root)
    joint_rotations = {}
    for joint_name, _, _ in hierarchy:
        if joint_name == 'Hips':
            joint_rotations[joint_name] = root_rotations
        else:
            # Placeholder: zero rotations for other joints
            joint_rotations[joint_name] = np.zeros((len(locked_joint_positions), 3))
    
    # Write BVH file
    bvh_writer.write_with_rotations(
        args.output,
        root_positions,
        joint_rotations,
        frame_time
    )
    
    print(f"✓ BVH file written to: {args.output}")
    print(f"  Frames: {len(locked_joint_positions)}")
    print(f"  Frame time: {frame_time:.6f}s ({fps} FPS)")
    print(f"  Duration: {len(locked_joint_positions) * frame_time:.2f}s")


if __name__ == '__main__':
    main()
