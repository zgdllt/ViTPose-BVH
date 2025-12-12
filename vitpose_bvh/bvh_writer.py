"""BVH file writer for motion capture data."""

import numpy as np
from typing import Dict, List, Tuple


class BVHWriter:
    """Writer for BVH (Biovision Hierarchy) files."""
    
    def __init__(self, skeleton_hierarchy: List[Tuple[str, str, List[str]]]):
        """Initialize BVH writer.
        
        Args:
            skeleton_hierarchy: List of (joint_name, parent_name, children) tuples
        """
        self.hierarchy = skeleton_hierarchy
        self.joint_offsets = {}
        self.joint_channels = {}
        self._setup_channels()
    
    def _setup_channels(self):
        """Setup channel configuration for each joint."""
        for joint_name, parent_name, _ in self.hierarchy:
            if parent_name is None:
                # Root joint has position + rotation
                self.joint_channels[joint_name] = ['Xposition', 'Yposition', 'Zposition', 'Zrotation', 'Xrotation', 'Yrotation']
            else:
                # Other joints have only rotation
                self.joint_channels[joint_name] = ['Zrotation', 'Xrotation', 'Yrotation']
    
    def set_offsets(self, joint_positions: Dict[str, np.ndarray]):
        """Set joint offsets from T-pose positions.
        
        Args:
            joint_positions: Dictionary mapping joint names to 3D positions
        """
        for joint_name, parent_name, _ in self.hierarchy:
            if parent_name is None:
                # Root offset is typically (0, 0, 0) or the initial position
                self.joint_offsets[joint_name] = np.array([0.0, 0.0, 0.0])
            elif parent_name in joint_positions and joint_name in joint_positions:
                # Offset relative to parent
                offset = joint_positions[joint_name] - joint_positions[parent_name]
                self.joint_offsets[joint_name] = offset
            else:
                # Fallback
                self.joint_offsets[joint_name] = np.array([0.0, 0.0, 0.0])
    
    def _write_joint(self, f, joint_name: str, parent_name: str, children: List[str], indent: int = 0):
        """Write a joint node in the hierarchy.
        
        Args:
            f: File object
            joint_name: Name of the joint
            parent_name: Name of parent joint
            children: List of child joint names
            indent: Indentation level
        """
        tab = '  ' * indent
        
        # Joint keyword
        if parent_name is None:
            f.write(f"{tab}ROOT {joint_name}\n")
        else:
            f.write(f"{tab}JOINT {joint_name}\n")
        
        f.write(f"{tab}{{\n")
        
        # Offset
        offset = self.joint_offsets.get(joint_name, np.array([0.0, 0.0, 0.0]))
        f.write(f"{tab}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n")
        
        # Channels
        channels = self.joint_channels.get(joint_name, [])
        num_channels = len(channels)
        channels_str = ' '.join(channels)
        f.write(f"{tab}  CHANNELS {num_channels} {channels_str}\n")
        
        # Children
        if children:
            for child_name in children:
                # Find child info in hierarchy
                for jn, pn, ch in self.hierarchy:
                    if jn == child_name:
                        self._write_joint(f, jn, pn, ch, indent + 1)
                        break
        else:
            # End site for leaf joints
            f.write(f"{tab}  End Site\n")
            f.write(f"{tab}  {{\n")
            # End site offset (small offset from parent)
            end_offset = np.array([0.0, 0.0, 0.1])
            f.write(f"{tab}    OFFSET {end_offset[0]:.6f} {end_offset[1]:.6f} {end_offset[2]:.6f}\n")
            f.write(f"{tab}  }}\n")
        
        f.write(f"{tab}}}\n")
    
    def write(self, filename: str, motion_data: List[Dict[str, np.ndarray]], frame_time: float):
        """Write BVH file.
        
        Args:
            filename: Output filename
            motion_data: List of frames, each frame is a dict mapping joint names to positions (3D)
            frame_time: Time between frames in seconds (1/fps)
        """
        with open(filename, 'w') as f:
            # Write HIERARCHY section
            f.write("HIERARCHY\n")
            
            # Find root joint
            root_joint = None
            for joint_name, parent_name, children in self.hierarchy:
                if parent_name is None:
                    root_joint = (joint_name, parent_name, children)
                    break
            
            if root_joint is None:
                raise ValueError("No root joint found in hierarchy")
            
            # Write hierarchy tree
            self._write_joint(f, root_joint[0], root_joint[1], root_joint[2], indent=0)
            
            # Write MOTION section
            f.write("MOTION\n")
            f.write(f"Frames: {len(motion_data)}\n")
            f.write(f"Frame Time: {frame_time:.6f}\n")
            
            # Write motion frames
            # For now, write zeros for rotations (placeholder)
            # This should be populated with actual rotation data
            for frame_data in motion_data:
                frame_values = []
                
                for joint_name, parent_name, _ in self.hierarchy:
                    channels = self.joint_channels.get(joint_name, [])
                    
                    for channel in channels:
                        if 'position' in channel.lower():
                            # Position channels (only for root)
                            axis = channel[0].lower()  # x, y, or z
                            if joint_name in frame_data:
                                pos = frame_data[joint_name]
                                if axis == 'x':
                                    frame_values.append(pos[0])
                                elif axis == 'y':
                                    frame_values.append(pos[1])
                                elif axis == 'z':
                                    frame_values.append(pos[2])
                            else:
                                frame_values.append(0.0)
                        else:
                            # Rotation channels - placeholder (will be filled by export script)
                            frame_values.append(0.0)
                
                # Write frame line
                frame_str = ' '.join(f'{v:.6f}' for v in frame_values)
                f.write(f"{frame_str}\n")
    
    def write_with_rotations(self, filename: str, 
                            root_positions: np.ndarray,
                            joint_rotations: Dict[str, np.ndarray],
                            frame_time: float):
        """Write BVH file with rotation data.
        
        Args:
            filename: Output filename
            root_positions: Root positions for each frame, shape (n_frames, 3)
            joint_rotations: Dictionary mapping joint names to rotation arrays, shape (n_frames, 3) in degrees
            frame_time: Time between frames in seconds (1/fps)
        """
        n_frames = len(root_positions)
        
        with open(filename, 'w') as f:
            # Write HIERARCHY section
            f.write("HIERARCHY\n")
            
            # Find root joint
            root_joint = None
            for joint_name, parent_name, children in self.hierarchy:
                if parent_name is None:
                    root_joint = (joint_name, parent_name, children)
                    break
            
            if root_joint is None:
                raise ValueError("No root joint found in hierarchy")
            
            # Write hierarchy tree
            self._write_joint(f, root_joint[0], root_joint[1], root_joint[2], indent=0)
            
            # Write MOTION section
            f.write("MOTION\n")
            f.write(f"Frames: {n_frames}\n")
            f.write(f"Frame Time: {frame_time:.6f}\n")
            
            # Write motion frames
            for frame_idx in range(n_frames):
                frame_values = []
                
                for joint_name, parent_name, _ in self.hierarchy:
                    channels = self.joint_channels.get(joint_name, [])
                    
                    for channel in channels:
                        if 'position' in channel.lower():
                            # Position channels (only for root)
                            axis = channel[0].lower()
                            pos = root_positions[frame_idx]
                            if axis == 'x':
                                frame_values.append(pos[0])
                            elif axis == 'y':
                                frame_values.append(pos[1])
                            elif axis == 'z':
                                frame_values.append(pos[2])
                        elif 'rotation' in channel.lower():
                            # Rotation channels
                            axis = channel[0].lower()
                            if joint_name in joint_rotations:
                                rot = joint_rotations[joint_name][frame_idx]
                                if axis == 'x':
                                    frame_values.append(rot[1])  # X rotation
                                elif axis == 'y':
                                    frame_values.append(rot[2])  # Y rotation
                                elif axis == 'z':
                                    frame_values.append(rot[0])  # Z rotation
                            else:
                                frame_values.append(0.0)
                
                # Write frame line
                frame_str = ' '.join(f'{v:.6f}' for v in frame_values)
                f.write(f"{frame_str}\n")
