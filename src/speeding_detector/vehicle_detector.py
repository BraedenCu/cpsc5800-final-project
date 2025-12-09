"""
Vehicle detection and feature extraction module.

Handles vehicle segmentation from point clouds and RGB images,
extracts BRISK features for tracking, and computes vehicle centroids.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Vehicle:
    """Represents a detected vehicle with its properties."""
    id: int
    centroid: np.ndarray  # 3D centroid in world coordinates
    point_cloud: np.ndarray  # Nx3 point cloud segment
    bbox: Tuple[int, int, int, int]  # Bounding box (x, y, w, h) in image space
    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray
    rgb_patch: Optional[np.ndarray] = None  # RGB image patch of the vehicle


class VehicleDetector:
    """Detects and extracts features from vehicles in camera frames."""

    def __init__(self, use_brisk: bool = True):
        """
        Initialize vehicle detector.

        Args:
            use_brisk: If True, use BRISK features. Otherwise use SIFT.
        """
        self.use_brisk = use_brisk

        # Initialize feature detector
        if use_brisk:
            self.feature_detector = cv2.BRISK_create()
        else:
            self.feature_detector = cv2.SIFT_create()

        # Feature matcher for tracking
        if use_brisk:
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        self.next_vehicle_id = 0

    def segment_vehicles(self, point_cloud: np.ndarray,
                        rgb_image: np.ndarray,
                        depth_image: np.ndarray,
                        ground_plane_threshold: float = 0.5,
                        min_points: int = 100) -> List[Vehicle]:
        """
        Segment vehicles from the point cloud and RGB image.

        Uses simple ground plane filtering and clustering to identify vehicles.
        In production, could use more sophisticated methods like DBSCAN or
        neural network segmentation.

        Args:
            point_cloud: Nx3 array of points in world coordinates
            rgb_image: HxWx3 RGB image
            depth_image: HxW depth image
            ground_plane_threshold: Height threshold to filter ground plane
            min_points: Minimum points required for a valid vehicle detection

        Returns:
            List of detected Vehicle objects
        """
        vehicles = []

        if point_cloud is None or len(point_cloud) == 0:
            return vehicles

        # Filter out ground plane (assume z-axis is height)
        # This is a simplified approach - could use RANSAC for plane fitting
        above_ground_mask = point_cloud[:, 2] > ground_plane_threshold

        if not np.any(above_ground_mask):
            return vehicles

        elevated_points = point_cloud[above_ground_mask]

        # Simple spatial clustering using grid-based approach
        # More sophisticated: use DBSCAN or connected components
        clusters = self._cluster_points(elevated_points, grid_size=0.5)

        for cluster_idx, cluster_points in enumerate(clusters):
            if len(cluster_points) < min_points:
                continue

            # Compute centroid
            centroid = np.mean(cluster_points, axis=0)

            # Project cluster to image space to extract RGB features
            bbox, rgb_patch = self._get_image_region(
                cluster_points, rgb_image, depth_image
            )

            if rgb_patch is not None:
                # Extract BRISK/SIFT features
                keypoints, descriptors = self.feature_detector.detectAndCompute(
                    rgb_patch, None
                )

                if descriptors is not None:
                    vehicle = Vehicle(
                        id=self.next_vehicle_id,
                        centroid=centroid,
                        point_cloud=cluster_points,
                        bbox=bbox,
                        keypoints=keypoints,
                        descriptors=descriptors,
                        rgb_patch=rgb_patch
                    )
                    vehicles.append(vehicle)
                    self.next_vehicle_id += 1

        return vehicles

    def _cluster_points(self, points: np.ndarray, grid_size: float = 0.5) -> List[np.ndarray]:
        """
        Simple grid-based spatial clustering of 3D points.

        Args:
            points: Nx3 array of 3D points
            grid_size: Size of grid cells in meters

        Returns:
            List of point clusters
        """
        if len(points) == 0:
            return []

        # Discretize points into grid cells
        grid_coords = (points / grid_size).astype(int)

        # Group points by grid cell
        unique_cells = {}
        for i, coord in enumerate(grid_coords):
            key = tuple(coord)
            if key not in unique_cells:
                unique_cells[key] = []
            unique_cells[key].append(i)

        # Merge adjacent cells using simple connected components
        clusters = []
        visited = set()

        for cell_key in unique_cells:
            if cell_key in visited:
                continue

            # BFS to find connected cluster
            cluster_indices = []
            queue = [cell_key]
            visited.add(cell_key)

            while queue:
                current = queue.pop(0)
                if current in unique_cells:
                    cluster_indices.extend(unique_cells[current])

                    # Check 26-connected neighbors
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            for dz in [-1, 0, 1]:
                                neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
                                if neighbor not in visited and neighbor in unique_cells:
                                    visited.add(neighbor)
                                    queue.append(neighbor)

            if cluster_indices:
                clusters.append(points[cluster_indices])

        return clusters

    def _get_image_region(self, points_3d: np.ndarray,
                         rgb_image: np.ndarray,
                         depth_image: np.ndarray) -> Tuple[Tuple[int, int, int, int], Optional[np.ndarray]]:
        """
        Project 3D points to image space and extract RGB patch.

        Args:
            points_3d: Nx3 array of 3D points in camera frame
            rgb_image: RGB image
            depth_image: Depth image

        Returns:
            Tuple of (bounding_box, rgb_patch)
        """
        # Simplified projection - assumes perspective projection
        # In production, use camera intrinsics for accurate projection
        if rgb_image is None or points_3d is None or len(points_3d) == 0:
            return (0, 0, 0, 0), None

        # Mock projection for this implementation
        # Real implementation would use camera intrinsics:
        # u = fx * x/z + cx
        # v = fy * y/z + cy

        h, w = rgb_image.shape[:2]
        bbox = (w // 4, h // 4, w // 2, h // 2)  # Placeholder
        rgb_patch = rgb_image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]

        return bbox, rgb_patch

    def match_vehicles(self, current_vehicles: List[Vehicle],
                      previous_vehicles: List[Vehicle],
                      max_distance: float = 2.0) -> List[Tuple[int, int]]:
        """
        Match vehicles between frames using feature descriptors and spatial proximity.

        Args:
            current_vehicles: List of vehicles in current frame
            previous_vehicles: List of vehicles in previous frame
            max_distance: Maximum centroid distance for matching (meters)

        Returns:
            List of (current_idx, previous_idx) matches
        """
        if not previous_vehicles or not current_vehicles:
            return []

        matches = []

        for curr_idx, curr_vehicle in enumerate(current_vehicles):
            best_match = None
            best_score = float('inf')

            for prev_idx, prev_vehicle in enumerate(previous_vehicles):
                # Spatial proximity check
                spatial_distance = np.linalg.norm(
                    curr_vehicle.centroid - prev_vehicle.centroid
                )

                if spatial_distance > max_distance:
                    continue

                # Feature descriptor similarity
                if curr_vehicle.descriptors is not None and prev_vehicle.descriptors is not None:
                    descriptor_matches = self.matcher.knnMatch(
                        curr_vehicle.descriptors,
                        prev_vehicle.descriptors,
                        k=2
                    )

                    # Apply ratio test (Lowe's ratio test)
                    good_matches = []
                    for match_pair in descriptor_matches:
                        if len(match_pair) == 2:
                            m, n = match_pair
                            if m.distance < 0.75 * n.distance:
                                good_matches.append(m)

                    # Combined score: weighted sum of spatial and feature distance
                    feature_score = len(good_matches) / max(len(curr_vehicle.descriptors), 1)
                    combined_score = spatial_distance - 2.0 * feature_score

                    if combined_score < best_score:
                        best_score = combined_score
                        best_match = prev_idx

            if best_match is not None:
                matches.append((curr_idx, best_match))

        return matches
