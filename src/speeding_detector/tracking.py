"""
Vehicle tracking module using Kalman filtering.

Implements state estimation for vehicle position and velocity to handle
measurement noise and occlusions. Each vehicle's state consists of
[x, y, z, vx, vy, vz] - 3D position and velocity.
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class VehicleTrack:
    """Represents a tracked vehicle with state estimation."""
    id: int
    kf: KalmanFilter
    last_seen_frame: int = 0
    frames_since_update: int = 0
    total_detections: int = 0
    velocity_history: List[float] = field(default_factory=list)
    max_velocity: float = 0.0

    def get_position(self) -> np.ndarray:
        """Get current estimated position [x, y, z]."""
        return self.kf.x[:3].flatten()

    def get_velocity(self) -> np.ndarray:
        """Get current estimated velocity [vx, vy, vz]."""
        return self.kf.x[3:].flatten()

    def get_speed(self) -> float:
        """Get current estimated speed (magnitude of velocity vector)."""
        velocity = self.get_velocity()
        return float(np.linalg.norm(velocity))


class VehicleTracker:
    """Tracks multiple vehicles using Kalman filtering."""

    def __init__(self,
                 dt: float = 0.033,  # ~30 FPS
                 process_noise: float = 0.1,
                 measurement_noise: float = 0.5,
                 max_frames_missing: int = 10):
        """
        Initialize vehicle tracker.

        Args:
            dt: Time step between frames in seconds
            process_noise: Process noise standard deviation
            measurement_noise: Measurement noise standard deviation
            max_frames_missing: Maximum frames before removing a track
        """
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.max_frames_missing = max_frames_missing

        self.tracks: Dict[int, VehicleTrack] = {}
        self.frame_count = 0
        self.next_track_id = 0

    def create_kalman_filter(self) -> KalmanFilter:
        """
        Create a Kalman filter for 3D position and velocity tracking.

        State vector: [x, y, z, vx, vy, vz]
        Measurement: [x, y, z]

        Returns:
            Initialized KalmanFilter
        """
        kf = KalmanFilter(dim_x=6, dim_z=3)

        # State transition matrix (constant velocity model)
        # x_new = x + vx*dt
        # y_new = y + vy*dt
        # z_new = z + vz*dt
        # vx_new = vx
        # vy_new = vy
        # vz_new = vz
        kf.F = np.array([
            [1, 0, 0, self.dt, 0,      0],
            [0, 1, 0, 0,      self.dt, 0],
            [0, 0, 1, 0,      0,      self.dt],
            [0, 0, 0, 1,      0,      0],
            [0, 0, 0, 0,      1,      0],
            [0, 0, 0, 0,      0,      1]
        ])

        # Measurement function (we only observe position)
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])

        # Process noise covariance
        # Higher noise in velocity than position
        q = self.process_noise
        kf.Q = np.eye(6) * q**2
        kf.Q[3:, 3:] *= 2  # Higher process noise for velocity

        # Measurement noise covariance
        r = self.measurement_noise
        kf.R = np.eye(3) * r**2

        # Initial state covariance (high uncertainty)
        kf.P = np.eye(6) * 10.0

        return kf

    def update(self, detections: List[np.ndarray], detection_ids: List[int]) -> None:
        """
        Update tracks with new detections.

        Args:
            detections: List of 3D centroids [x, y, z]
            detection_ids: List of vehicle IDs from detector
        """
        self.frame_count += 1

        # Predict step for all existing tracks
        for track in self.tracks.values():
            track.kf.predict()
            track.frames_since_update += 1

        # Update step for matched detections
        for detection, det_id in zip(detections, detection_ids):
            if det_id in self.tracks:
                # Update existing track
                track = self.tracks[det_id]
                track.kf.update(detection)
                track.last_seen_frame = self.frame_count
                track.frames_since_update = 0
                track.total_detections += 1

                # Update velocity history
                speed = track.get_speed()
                track.velocity_history.append(speed)
                track.max_velocity = max(track.max_velocity, speed)
            else:
                # Create new track
                track = self._create_new_track(detection, det_id)
                self.tracks[det_id] = track

        # Remove stale tracks
        self._remove_stale_tracks()

    def _create_new_track(self, position: np.ndarray, vehicle_id: int) -> VehicleTrack:
        """
        Create a new vehicle track.

        Args:
            position: Initial 3D position [x, y, z]
            vehicle_id: Vehicle ID from detector

        Returns:
            New VehicleTrack
        """
        kf = self.create_kalman_filter()

        # Initialize state with position, zero velocity
        kf.x = np.array([
            position[0], position[1], position[2],
            0.0, 0.0, 0.0
        ]).reshape((6, 1))

        track = VehicleTrack(
            id=vehicle_id,
            kf=kf,
            last_seen_frame=self.frame_count,
            total_detections=1
        )

        return track

    def _remove_stale_tracks(self) -> None:
        """Remove tracks that haven't been updated recently."""
        to_remove = []
        for track_id, track in self.tracks.items():
            if track.frames_since_update > self.max_frames_missing:
                to_remove.append(track_id)

        for track_id in to_remove:
            del self.tracks[track_id]

    def get_track(self, vehicle_id: int) -> Optional[VehicleTrack]:
        """Get track by vehicle ID."""
        return self.tracks.get(vehicle_id)

    def get_all_tracks(self) -> List[VehicleTrack]:
        """Get all active tracks."""
        return list(self.tracks.values())

    def get_active_tracks(self, max_frames_missing: int = 3) -> List[VehicleTrack]:
        """
        Get recently updated tracks.

        Args:
            max_frames_missing: Maximum frames since last update

        Returns:
            List of active tracks
        """
        return [
            track for track in self.tracks.values()
            if track.frames_since_update <= max_frames_missing
        ]
