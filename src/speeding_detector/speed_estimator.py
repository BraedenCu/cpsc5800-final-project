"""
Speed estimation and metadata capture module.

Computes vehicle speeds from tracked positions and captures relevant
information when speeding violations are detected.
"""

import numpy as np
import cv2
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
import json


@dataclass
class SpeedingViolation:
    """Records information about a speeding violation."""
    vehicle_id: int
    timestamp: str
    estimated_speed_mps: float  # meters per second
    estimated_speed_mph: float  # miles per hour
    speed_limit_mph: float
    position: tuple  # (x, y, z) in world coordinates
    frame_number: int
    image_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class SpeedEstimator:
    """Estimates vehicle speeds and detects speeding violations."""

    # Conversion factor: meters per second to miles per hour
    MPS_TO_MPH = 2.23694

    def __init__(self,
                 speed_limit_mph: float = 25.0,
                 output_dir: str = "./violations",
                 save_images: bool = True):
        """
        Initialize speed estimator.

        Args:
            speed_limit_mph: Speed limit in miles per hour
            output_dir: Directory to save violation records and images
            save_images: Whether to save images of violations
        """
        self.speed_limit_mph = speed_limit_mph
        self.speed_limit_mps = speed_limit_mph / self.MPS_TO_MPH
        self.output_dir = Path(output_dir)
        self.save_images = save_images

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.output_dir / "images"
        if self.save_images:
            self.images_dir.mkdir(exist_ok=True)

        self.violations: list[SpeedingViolation] = []
        self.detected_vehicle_ids: set = set()

    def compute_speed(self,
                     position_current: np.ndarray,
                     position_previous: np.ndarray,
                     dt: float) -> float:
        """
        Compute speed from successive positions.

        As specified in the proposal:
        v = ||P_{t+1} - P_t|| / Δt

        Args:
            position_current: Current 3D position [x, y, z]
            position_previous: Previous 3D position [x, y, z]
            dt: Time between measurements in seconds

        Returns:
            Speed in meters per second
        """
        if dt <= 0:
            return 0.0

        displacement = np.linalg.norm(position_current - position_previous)
        speed = displacement / dt

        return speed

    def check_speeding(self,
                      vehicle_id: int,
                      speed_mps: float,
                      position: np.ndarray,
                      frame_number: int,
                      rgb_image: Optional[np.ndarray] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> Optional[SpeedingViolation]:
        """
        Check if vehicle is speeding and record violation.

        Args:
            vehicle_id: Vehicle ID
            speed_mps: Estimated speed in meters per second
            position: 3D position [x, y, z]
            frame_number: Current frame number
            rgb_image: RGB image of the vehicle (optional)
            metadata: Additional metadata (optional)

        Returns:
            SpeedingViolation if speeding detected, None otherwise
        """
        # Avoid duplicate violations for same vehicle
        if vehicle_id in self.detected_vehicle_ids:
            return None

        speed_mph = speed_mps * self.MPS_TO_MPH

        if speed_mph > self.speed_limit_mph:
            # Save image if available
            image_path = None
            if self.save_images and rgb_image is not None:
                image_filename = f"violation_{vehicle_id}_{frame_number}.jpg"
                image_path = str(self.images_dir / image_filename)
                cv2.imwrite(image_path, rgb_image)

            # Create violation record
            violation = SpeedingViolation(
                vehicle_id=vehicle_id,
                timestamp=datetime.now().isoformat(),
                estimated_speed_mps=speed_mps,
                estimated_speed_mph=speed_mph,
                speed_limit_mph=self.speed_limit_mph,
                position=tuple(position.tolist()),
                frame_number=frame_number,
                image_path=image_path,
                metadata=metadata
            )

            self.violations.append(violation)
            self.detected_vehicle_ids.add(vehicle_id)

            return violation

        return None

    def save_violations(self, filename: str = "violations.json") -> None:
        """
        Save all violations to JSON file.

        Args:
            filename: Output filename
        """
        output_path = self.output_dir / filename
        violations_data = [v.to_dict() for v in self.violations]

        with open(output_path, 'w') as f:
            json.dump(violations_data, f, indent=2)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about detected violations.

        Returns:
            Dictionary with violation statistics
        """
        if not self.violations:
            return {
                'total_violations': 0,
                'average_speed_mph': 0.0,
                'max_speed_mph': 0.0,
                'min_speed_mph': 0.0
            }

        speeds = [v.estimated_speed_mph for v in self.violations]

        return {
            'total_violations': len(self.violations),
            'average_speed_mph': float(np.mean(speeds)),
            'max_speed_mph': float(np.max(speeds)),
            'min_speed_mph': float(np.min(speeds)),
            'speed_limit_mph': self.speed_limit_mph,
            'average_overage_mph': float(np.mean(speeds) - self.speed_limit_mph)
        }

    def generate_report(self) -> str:
        """
        Generate a human-readable report of violations.

        Returns:
            Report string
        """
        stats = self.get_statistics()

        report = []
        report.append("=" * 60)
        report.append("SPEEDING VIOLATION REPORT")
        report.append("=" * 60)
        report.append(f"Speed Limit: {self.speed_limit_mph:.1f} mph")
        report.append(f"Total Violations: {stats['total_violations']}")
        report.append("")

        if stats['total_violations'] > 0:
            report.append(f"Average Speed: {stats['average_speed_mph']:.1f} mph")
            report.append(f"Maximum Speed: {stats['max_speed_mph']:.1f} mph")
            report.append(f"Minimum Speed: {stats['min_speed_mph']:.1f} mph")
            report.append(f"Average Overage: {stats['average_overage_mph']:.1f} mph")
            report.append("")
            report.append("Individual Violations:")
            report.append("-" * 60)

            for i, violation in enumerate(self.violations, 1):
                report.append(f"{i}. Vehicle ID: {violation.vehicle_id}")
                report.append(f"   Time: {violation.timestamp}")
                report.append(f"   Speed: {violation.estimated_speed_mph:.1f} mph")
                report.append(f"   Overage: {violation.estimated_speed_mph - self.speed_limit_mph:.1f} mph")
                if violation.image_path:
                    report.append(f"   Image: {violation.image_path}")
                report.append("")

        report.append("=" * 60)

        return "\n".join(report)
