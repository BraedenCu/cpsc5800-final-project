"""
Forward kinematics module for UFACTORY Lite6 robotic arm.

Implements the Denavit-Hartenberg (DH) convention to compute the end effector pose
from joint angles. This allows us to determine the RealSense camera position and
orientation in the world (robot base) coordinate frame.
"""

import numpy as np
from typing import List, Tuple


class DHParameter:
    """Represents a single link's Denavit-Hartenberg parameters."""

    def __init__(self, theta: float, d: float, a: float, alpha: float):
        """
        Initialize DH parameters for a joint.

        Args:
            theta: Joint angle (radians)
            d: Offset along previous z-axis to the common normal
            a: Length of the common normal
            alpha: Angle about common normal from previous z to current z
        """
        self.theta = theta
        self.d = d
        self.a = a
        self.alpha = alpha

    def transformation_matrix(self) -> np.ndarray:
        """
        Compute the homogeneous transformation matrix for this joint.

        Uses the modified DH convention as specified in the proposal:
        T_i = [cos(θ_i)  -sin(θ_i)cos(α_i)   sin(θ_i)sin(α_i)   a_i*cos(θ_i)]
              [sin(θ_i)   cos(θ_i)cos(α_i)  -cos(θ_i)sin(α_i)   a_i*sin(θ_i)]
              [0          sin(α_i)            cos(α_i)            d_i         ]
              [0          0                   0                   1           ]

        Returns:
            4x4 homogeneous transformation matrix
        """
        ct = np.cos(self.theta)
        st = np.sin(self.theta)
        ca = np.cos(self.alpha)
        sa = np.sin(self.alpha)

        T = np.array([
            [ct, -st * ca,  st * sa, self.a * ct],
            [st,  ct * ca, -ct * sa, self.a * st],
            [0,   sa,       ca,      self.d],
            [0,   0,        0,       1]
        ])

        return T


class Lite6ForwardKinematics:
    """Forward kinematics for UFACTORY Lite6 6-DOF robotic arm."""

    def __init__(self, dh_params_static: List[Tuple[float, float, float]] = None):
        """
        Initialize the forward kinematics solver.

        Args:
            dh_params_static: List of static DH parameters (d, a, alpha) for each joint.
                            If None, uses default Lite6 parameters.
        """
        if dh_params_static is None:
            # Default DH parameters for UFACTORY Lite6
            # These are placeholder values - should be replaced with actual Lite6 specs
            # Format: (d, a, alpha) for each of the 6 joints
            self.dh_params_static = [
                (0.2435, 0.0,    np.pi/2),  # Joint 1
                (0.0,    0.2073, 0.0),      # Joint 2
                (0.0,    0.2073, 0.0),      # Joint 3
                (0.1038, 0.0,    np.pi/2),  # Joint 4
                (0.0956, 0.0,    -np.pi/2), # Joint 5
                (0.1072, 0.0,    0.0)       # Joint 6
            ]
        else:
            self.dh_params_static = dh_params_static

    def compute_end_effector_pose(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute end effector pose T_{0->6} from joint angles.

        The end effector pose is computed by multiplying the individual link
        transformation matrices:
        T_{0->6} = T_1 * T_2 * T_3 * T_4 * T_5 * T_6

        Args:
            joint_angles: Array of 6 joint angles in radians

        Returns:
            4x4 homogeneous transformation matrix representing end effector pose
        """
        if len(joint_angles) != 6:
            raise ValueError(f"Expected 6 joint angles, got {len(joint_angles)}")

        # Initialize with identity matrix
        T_cumulative = np.eye(4)

        # Multiply transformation matrices for each joint
        for i, (theta, (d, a, alpha)) in enumerate(zip(joint_angles, self.dh_params_static)):
            dh_param = DHParameter(theta, d, a, alpha)
            T_i = dh_param.transformation_matrix()
            T_cumulative = T_cumulative @ T_i

        return T_cumulative

    def get_camera_pose(self, joint_angles: np.ndarray,
                       camera_offset: np.ndarray = None) -> np.ndarray:
        """
        Get the camera pose in world coordinates, accounting for mounting offset.

        Args:
            joint_angles: Array of 6 joint angles in radians
            camera_offset: 4x4 transformation from end effector to camera center.
                         If None, assumes camera is at end effector center.

        Returns:
            4x4 transformation matrix from world to camera frame
        """
        T_base_to_ee = self.compute_end_effector_pose(joint_angles)

        if camera_offset is not None:
            T_base_to_camera = T_base_to_ee @ camera_offset
        else:
            T_base_to_camera = T_base_to_ee

        return T_base_to_camera

    def extract_position_and_orientation(self, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract position and orientation from transformation matrix.

        Args:
            T: 4x4 homogeneous transformation matrix

        Returns:
            Tuple of (position, rotation_matrix) where:
                - position is a 3D vector [x, y, z]
                - rotation_matrix is the 3x3 rotation submatrix
        """
        position = T[:3, 3]
        rotation = T[:3, :3]
        return position, rotation
