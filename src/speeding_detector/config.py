"""
Configuration parameters for the speeding detection system.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class CameraConfig:
    """RealSense camera configuration."""
    width: int = 640
    height: int = 480
    fps: int = 30
    spatial_filter_sigma: float = 2.0
    temporal_filter_alpha: float = 0.7


@dataclass
class DetectionConfig:
    """Vehicle detection configuration."""
    use_brisk: bool = True  # Use BRISK (faster) vs SIFT (more accurate)
    ground_plane_threshold: float = 0.5  # meters
    min_points_per_vehicle: int = 100
    max_matching_distance: float = 2.0  # meters
    clustering_grid_size: float = 0.5  # meters


@dataclass
class TrackingConfig:
    """Kalman filter tracking configuration."""
    process_noise: float = 0.1
    measurement_noise: float = 0.5
    max_frames_missing: int = 10


@dataclass
class SpeedConfig:
    """Speed estimation configuration."""
    speed_limit_mph: float = 25.0
    save_violation_images: bool = True
    output_directory: str = "./violations"


@dataclass
class RobotConfig:
    """UFACTORY Lite6 robot configuration."""
    # DH parameters: (d, a, alpha) for each of 6 joints
    # These are placeholder values - replace with actual Lite6 specifications
    dh_parameters: Tuple[Tuple[float, float, float], ...] = (
        (0.2435, 0.0,    1.5708),  # Joint 1 (pi/2 rad)
        (0.0,    0.2073, 0.0),     # Joint 2
        (0.0,    0.2073, 0.0),     # Joint 3
        (0.1038, 0.0,    1.5708),  # Joint 4 (pi/2 rad)
        (0.0956, 0.0,   -1.5708),  # Joint 5 (-pi/2 rad)
        (0.1072, 0.0,    0.0)      # Joint 6
    )

    # Camera mounting offset from end effector (if needed)
    camera_offset_x: float = 0.0
    camera_offset_y: float = 0.0
    camera_offset_z: float = 0.05  # 5cm offset


@dataclass
class SystemConfig:
    """Complete system configuration."""
    camera: CameraConfig = CameraConfig()
    detection: DetectionConfig = DetectionConfig()
    tracking: TrackingConfig = TrackingConfig()
    speed: SpeedConfig = SpeedConfig()
    robot: RobotConfig = RobotConfig()

    def __post_init__(self):
        """Validate configuration."""
        assert self.camera.fps > 0, "FPS must be positive"
        assert self.speed.speed_limit_mph > 0, "Speed limit must be positive"
        assert self.detection.min_points_per_vehicle > 0, "Min points must be positive"


# Default configuration
DEFAULT_CONFIG = SystemConfig()
