"""
Camera and point cloud transformation utilities.

Handles transformation of point clouds from camera coordinate frame to world
coordinate frame using the transformation matrix computed from forward kinematics.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.ndimage import gaussian_filter


class PointCloudTransformer:
    """Transforms point clouds from camera frame to world frame."""

    @staticmethod
    def transform_point_cloud(points_camera: np.ndarray,
                             T_world_to_camera: np.ndarray) -> np.ndarray:
        """
        Transform point cloud from camera frame to world frame.

        As specified in the proposal:
        p^world = T_{0->6} * p^cam

        Args:
            points_camera: Nx3 array of 3D points in camera frame
            T_world_to_camera: 4x4 transformation matrix from forward kinematics

        Returns:
            Nx3 array of 3D points in world frame
        """
        # Convert to homogeneous coordinates (Nx4)
        N = points_camera.shape[0]
        ones = np.ones((N, 1))
        points_camera_homogeneous = np.hstack([points_camera, ones])

        # Apply transformation: (4x4) @ (4xN) = (4xN)
        points_world_homogeneous = (T_world_to_camera @ points_camera_homogeneous.T).T

        # Convert back to 3D coordinates
        points_world = points_world_homogeneous[:, :3]

        return points_world

    @staticmethod
    def filter_depth_data(depth_image: np.ndarray,
                         spatial_sigma: float = 2.0,
                         temporal_alpha: float = 0.7,
                         prev_depth: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply spatial and temporal filtering to reduce depth noise.

        Args:
            depth_image: Current depth image
            spatial_sigma: Sigma for spatial Gaussian filter
            temporal_alpha: Temporal smoothing factor (0=all current, 1=all previous)
            prev_depth: Previous filtered depth image for temporal filtering

        Returns:
            Filtered depth image
        """
        # Spatial filtering with Gaussian
        filtered = gaussian_filter(depth_image, sigma=spatial_sigma)

        # Temporal filtering if previous frame available
        if prev_depth is not None:
            filtered = temporal_alpha * prev_depth + (1 - temporal_alpha) * filtered

        return filtered


class RealSenseInterface:
    """
    Interface for Intel RealSense camera.

    Note: This is a mock interface since pyrealsense2 doesn't support macOS ARM.
    On the actual x86_64 Linux system with the robot, this would use pyrealsense2.
    """

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        """
        Initialize RealSense camera interface.

        Args:
            width: Frame width
            height: Frame height
            fps: Frames per second
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.intrinsics = None
        self.prev_depth = None

        # On actual hardware, would initialize:
        # import pyrealsense2 as rs
        # self.pipeline = rs.pipeline()
        # self.config = rs.config()
        # self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        # self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    def start(self):
        """Start the camera stream."""
        # On actual hardware: self.pipeline.start(self.config)
        pass

    def stop(self):
        """Stop the camera stream."""
        # On actual hardware: self.pipeline.stop()
        pass

    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Capture a frame from the RealSense camera.

        Returns:
            Tuple of (rgb_image, depth_image, point_cloud):
                - rgb_image: HxWx3 RGB image
                - depth_image: HxW depth image in meters
                - point_cloud: Nx3 array of 3D points in camera frame
        """
        # On actual hardware, would:
        # frames = self.pipeline.wait_for_frames()
        # depth_frame = frames.get_depth_frame()
        # color_frame = frames.get_color_frame()
        #
        # Apply filters
        # depth_image = np.asanyarray(depth_frame.get_data()) * depth_frame.get_units()
        # rgb_image = np.asanyarray(color_frame.get_data())
        #
        # Generate point cloud
        # pc = rs.pointcloud()
        # points = pc.calculate(depth_frame)
        # point_cloud = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)

        # Mock return for development
        return None, None, None

    def get_camera_intrinsics(self) -> dict:
        """
        Get camera intrinsic parameters.

        Returns:
            Dictionary with fx, fy, cx, cy, distortion coefficients
        """
        # On actual hardware:
        # profile = self.pipeline.get_active_profile()
        # color_profile = profile.get_stream(rs.stream.color)
        # intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

        # Mock intrinsics
        return {
            'fx': 615.0,
            'fy': 615.0,
            'cx': self.width / 2,
            'cy': self.height / 2,
            'distortion': [0.0, 0.0, 0.0, 0.0, 0.0]
        }
