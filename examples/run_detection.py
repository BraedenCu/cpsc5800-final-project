"""
Example script to run the speeding detection system.

This demonstrates how to use the pipeline for a 10-minute monitoring session
as specified in the project proposal.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from speeding_detector.pipeline import SpeedingDetectionPipeline
from speeding_detector.config import SystemConfig


def get_joint_angles_from_robot():
    """
    Get current joint angles from the UFACTORY Lite6 robot.

    On actual hardware, this would interface with the robot controller:
    - Use the UFACTORY SDK to read current joint positions
    - Convert to radians if necessary
    - Return as numpy array

    Returns:
        Array of 6 joint angles in radians
    """
    # Mock implementation for development
    # In production, would connect to robot controller:
    # from xarm.wrapper import XArmAPI
    # arm = XArmAPI('192.168.1.xxx')
    # positions = arm.get_servo_angle()
    # return np.deg2rad(positions[1])  # Convert to radians

    # For now, return static angles (camera pointing down the street)
    return np.array([0.0, -0.5, 0.5, 0.0, 0.5, 0.0])


def main():
    """Run the speeding detection system for 10 minutes."""
    # Load configuration
    config = SystemConfig()

    print("=" * 60)
    print("SPEEDING VEHICLE DETECTION SYSTEM")
    print("=" * 60)
    print(f"Speed Limit: {config.speed.speed_limit_mph} mph")
    print(f"Detection Duration: 10 minutes")
    print(f"Camera FPS: {config.camera.fps}")
    print(f"Using BRISK features: {config.detection.use_brisk}")
    print(f"Output Directory: {config.speed.output_directory}")
    print("=" * 60)
    print()

    # Initialize pipeline
    pipeline = SpeedingDetectionPipeline(
        speed_limit_mph=config.speed.speed_limit_mph,
        fps=config.camera.fps,
        output_dir=config.speed.output_directory,
        use_brisk=config.detection.use_brisk
    )

    try:
        # Run for 10 minutes (600 seconds) as specified in proposal
        pipeline.run(
            duration_seconds=600.0,
            get_joint_angles_fn=get_joint_angles_from_robot
        )

    except KeyboardInterrupt:
        print("\n\nDetection interrupted by user")
        pipeline.stop()

    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        pipeline.stop()

    print("\nDetection complete. Check output directory for results.")


if __name__ == "__main__":
    main()
