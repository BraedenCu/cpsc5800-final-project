"""
Main pipeline for speeding vehicle detection system.

Integrates all components to implement the algorithm from the proposal:
1. Segment vehicles in the global point cloud using forward kinematics
2. Extract BRISK features for each vehicle
3. Match features and centroids to vehicles in previous frames
4. Update tracked states using Kalman filtering
5. Capture vehicle information and metadata if speed exceeds threshold
"""

import numpy as np
import time
from typing import Optional, List
from pathlib import Path

from .forward_kinematics import Lite6ForwardKinematics
from .camera_transform import PointCloudTransformer, RealSenseInterface
from .vehicle_detector import VehicleDetector, Vehicle
from .tracking import VehicleTracker, VehicleTrack
from .speed_estimator import SpeedEstimator


class SpeedingDetectionPipeline:
    """Main pipeline for detecting and recording speeding vehicles."""

    def __init__(self,
                 speed_limit_mph: float = 25.0,
                 fps: int = 30,
                 output_dir: str = "./violations",
                 use_brisk: bool = True):
        """
        Initialize the speeding detection pipeline.

        Args:
            speed_limit_mph: Speed limit for the monitored area
            fps: Camera frame rate
            output_dir: Directory to save violation records
            use_brisk: Use BRISK features (faster) vs SIFT (more accurate)
        """
        # Initialize components
        self.kinematics = Lite6ForwardKinematics()
        self.transformer = PointCloudTransformer()
        self.camera = RealSenseInterface(fps=fps)
        self.detector = VehicleDetector(use_brisk=use_brisk)
        self.tracker = VehicleTracker(dt=1.0/fps)
        self.speed_estimator = SpeedEstimator(
            speed_limit_mph=speed_limit_mph,
            output_dir=output_dir
        )

        self.fps = fps
        self.frame_count = 0
        self.total_vehicles_detected = 0
        self.running = False

        # Previous frame data for tracking
        self.previous_vehicles: List[Vehicle] = []

    def start(self) -> None:
        """Start the camera stream."""
        self.camera.start()
        self.running = True
        print(f"Pipeline started. Monitoring speed limit: {self.speed_estimator.speed_limit_mph} mph")

    def stop(self) -> None:
        """Stop the camera stream and save results."""
        self.running = False
        self.camera.stop()
        self.speed_estimator.save_violations()
        print("Pipeline stopped. Results saved.")

    def process_frame(self, joint_angles: np.ndarray) -> dict:
        """
        Process a single frame through the complete pipeline.

        Implements the algorithm steps from the proposal:
        1. Segment vehicles in the global point cloud
        2. Extract features for each vehicle using BRISK
        3. Match features and centroids to vehicles in previous frames
        4. Update tracked states using Kalman filtering
        5. Capture relevant vehicle information if speed exceeds threshold

        Args:
            joint_angles: Current robot joint angles [6 values in radians]

        Returns:
            Dictionary with frame processing results
        """
        self.frame_count += 1

        # Step 1: Get camera pose using forward kinematics
        T_world_to_camera = self.kinematics.get_camera_pose(joint_angles)

        # Get frame from RealSense
        rgb_image, depth_image, point_cloud_camera = self.camera.get_frame()

        # Mock data for development (since RealSense not available on macOS ARM)
        if rgb_image is None:
            return self._mock_frame_result()

        # Transform point cloud to world coordinates
        point_cloud_world = self.transformer.transform_point_cloud(
            point_cloud_camera, T_world_to_camera
        )

        # Step 2: Segment vehicles and extract BRISK features
        current_vehicles = self.detector.segment_vehicles(
            point_cloud_world, rgb_image, depth_image
        )

        if not current_vehicles:
            return {
                'frame': self.frame_count,
                'vehicles_detected': 0,
                'violations': 0
            }

        # Step 3: Match features to vehicles in previous frame
        matches = []
        if self.previous_vehicles:
            matches = self.detector.match_vehicles(
                current_vehicles, self.previous_vehicles
            )

        # Map matched vehicles to previous IDs
        vehicle_id_map = {}
        for curr_idx, prev_idx in matches:
            vehicle_id_map[current_vehicles[curr_idx].id] = \
                self.previous_vehicles[prev_idx].id

        # Assign consistent IDs to matched vehicles
        for vehicle in current_vehicles:
            if vehicle.id in vehicle_id_map:
                vehicle.id = vehicle_id_map[vehicle.id]

        # Step 4: Update Kalman filter tracking
        detections = [v.centroid for v in current_vehicles]
        detection_ids = [v.id for v in current_vehicles]
        self.tracker.update(detections, detection_ids)

        # Step 5: Check for speeding violations
        violations_this_frame = 0
        for vehicle in current_vehicles:
            track = self.tracker.get_track(vehicle.id)
            if track is not None and track.total_detections >= 2:
                speed = track.get_speed()

                # Capture violation if speed exceeds limit
                violation = self.speed_estimator.check_speeding(
                    vehicle_id=vehicle.id,
                    speed_mps=speed,
                    position=track.get_position(),
                    frame_number=self.frame_count,
                    rgb_image=vehicle.rgb_patch,
                    metadata={
                        'total_detections': track.total_detections,
                        'velocity_vector': track.get_velocity().tolist()
                    }
                )

                if violation is not None:
                    violations_this_frame += 1
                    print(f"⚠️  VIOLATION: Vehicle {vehicle.id} - "
                          f"{violation.estimated_speed_mph:.1f} mph "
                          f"(+{violation.estimated_speed_mph - violation.speed_limit_mph:.1f} mph)")

        # Update state for next frame
        self.previous_vehicles = current_vehicles
        self.total_vehicles_detected = max(self.total_vehicles_detected,
                                          max(detection_ids, default=0))

        return {
            'frame': self.frame_count,
            'vehicles_detected': len(current_vehicles),
            'violations': violations_this_frame,
            'total_violations': len(self.speed_estimator.violations)
        }

    def _mock_frame_result(self) -> dict:
        """Generate mock result for development without hardware."""
        return {
            'frame': self.frame_count,
            'vehicles_detected': 0,
            'violations': 0,
            'note': 'Mock data - RealSense not available'
        }

    def run(self,
            duration_seconds: float = 600.0,
            get_joint_angles_fn=None) -> None:
        """
        Run the pipeline for a specified duration.

        Args:
            duration_seconds: Duration to run (default 600s = 10 minutes)
            get_joint_angles_fn: Function that returns current joint angles.
                               If None, uses mock joint angles.
        """
        self.start()

        start_time = time.time()
        frame_interval = 1.0 / self.fps

        print(f"Running detection for {duration_seconds/60:.1f} minutes...")
        print(f"Target FPS: {self.fps}")
        print("-" * 60)

        try:
            while self.running and (time.time() - start_time) < duration_seconds:
                frame_start = time.time()

                # Get current joint angles
                if get_joint_angles_fn is not None:
                    joint_angles = get_joint_angles_fn()
                else:
                    # Mock static joint angles for development
                    joint_angles = np.array([0.0, -0.5, 0.5, 0.0, 0.5, 0.0])

                # Process frame
                result = self.process_frame(joint_angles)

                # Display progress every 30 frames (~1 second at 30 fps)
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    print(f"[{elapsed:.1f}s] Frame {self.frame_count}: "
                          f"{result['vehicles_detected']} vehicles, "
                          f"{result['total_violations']} total violations")

                # Maintain frame rate
                frame_elapsed = time.time() - frame_start
                sleep_time = max(0, frame_interval - frame_elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nStopped by user")

        finally:
            self.stop()
            self._print_final_report()

    def _print_final_report(self) -> None:
        """Print final statistics and report."""
        print("\n" + "=" * 60)
        print("FINAL REPORT")
        print("=" * 60)
        print(f"Total frames processed: {self.frame_count}")
        print(f"Total unique vehicles detected: {self.total_vehicles_detected}")
        print(f"Total speeding violations: {len(self.speed_estimator.violations)}")
        print()
        print(self.speed_estimator.generate_report())


def main():
    """Main entry point for the speeding detection system."""
    # Initialize pipeline with 25 mph speed limit (typical for residential areas)
    pipeline = SpeedingDetectionPipeline(
        speed_limit_mph=25.0,
        fps=30,
        output_dir="./violations",
        use_brisk=True  # BRISK is faster than SIFT
    )

    # Run for 10 minutes as specified in the proposal
    pipeline.run(duration_seconds=600.0)


if __name__ == "__main__":
    main()
