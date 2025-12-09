# Speeding Vehicle Detection System

A computer vision system for detecting and recording speeding vehicles using a UFACTORY Lite6 robotic arm with Intel RealSense depth camera.

## Project Overview

This project implements the algorithm described in the CPSC5800 project proposal. It monitors vehicles passing through the Hillhouse and Grove Avenue intersection, tracks their speeds using 3D point cloud data, and captures information about vehicles exceeding the speed limit.

## System Architecture

The system consists of several key components:

### 1. Forward Kinematics ([forward_kinematics.py](src/speeding_detector/forward_kinematics.py))
- Implements Denavit-Hartenberg (DH) parameter transformations
- Computes end effector pose TÄãÜ from joint angles
- Transforms camera coordinates to world frame

### 2. Point Cloud Transformation ([camera_transform.py](src/speeding_detector/camera_transform.py))
- Interfaces with Intel RealSense depth camera
- Applies spatial and temporal filtering to reduce noise
- Transforms point clouds from camera frame to world frame

### 3. Vehicle Detection ([vehicle_detector.py](src/speeding_detector/vehicle_detector.py))
- Segments vehicles from point clouds
- Extracts BRISK/SIFT features for tracking
- Matches vehicles across successive frames

### 4. Kalman Filtering ([tracking.py](src/speeding_detector/tracking.py))
- Tracks vehicle state (position + velocity) over time
- Handles measurement noise and temporary occlusions
- Maintains consistent vehicle IDs across frames

### 5. Speed Estimation ([speed_estimator.py](src/speeding_detector/speed_estimator.py))
- Computes velocity: v = ||P_{t+1} - P_t|| / ît
- Detects speeding violations
- Captures snapshots and metadata

### 6. Main Pipeline ([pipeline.py](src/speeding_detector/pipeline.py))
- Integrates all components
- Implements the complete algorithm from the proposal
- Manages the 10-minute monitoring session

## Algorithm

For each frame, the system performs the following steps:

1. **Segment vehicles** in the global point cloud, performing forward kinematics to convert coordinates from the RealSense frame into the global frame
2. **Extract features** for each vehicle using BRISK
3. **Match features** and centroids to vehicles in previous frames using spatial proximity and descriptor similarity
4. **Update tracked states** using Kalman filtering
5. **Capture relevant vehicle information** and metadata if vehicle speed exceeds threshold

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

### Prerequisites
- Python 3.8+
- uv package manager
- UFACTORY Lite6 robotic arm (for hardware deployment)
- Intel RealSense depth camera (for hardware deployment)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd cpsc5800-final-project

# Install dependencies with uv
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

### Hardware Dependencies

**Note:** The `pyrealsense2` package is required for actual hardware deployment but is not available for macOS ARM. On the deployment system (x86_64 Linux), install it separately:

```bash
# On x86_64 Linux deployment system
uv pip install pyrealsense2
```

## Usage

### Basic Usage

Run a 10-minute monitoring session:

```bash
python examples/run_detection.py
```

### Custom Configuration

```python
from speeding_detector.pipeline import SpeedingDetectionPipeline
from speeding_detector.config import SystemConfig

# Customize configuration
config = SystemConfig()
config.speed.speed_limit_mph = 30.0
config.camera.fps = 60

# Create pipeline
pipeline = SpeedingDetectionPipeline(
    speed_limit_mph=config.speed.speed_limit_mph,
    fps=config.camera.fps,
    output_dir="./my_violations"
)

# Run detection
pipeline.run(duration_seconds=600.0)
```

### Hardware Integration

To integrate with actual UFACTORY Lite6 hardware:

```python
from xarm.wrapper import XArmAPI
import numpy as np

# Connect to robot
arm = XArmAPI('192.168.1.xxx')  # Robot IP address

def get_joint_angles():
    """Read current joint angles from robot."""
    positions = arm.get_servo_angle()
    return np.deg2rad(positions[1])  # Convert to radians

# Run with real hardware
pipeline.run(
    duration_seconds=600.0,
    get_joint_angles_fn=get_joint_angles
)
```

## Output

The system generates the following outputs in the specified directory (default: `./violations`):

### Files
- `violations.json` - JSON file with all violation records
- `images/` - Directory containing snapshots of speeding vehicles
  - Format: `violation_{vehicle_id}_{frame_number}.jpg`

### Violation Record Format

```json
{
  "vehicle_id": 42,
  "timestamp": "2025-11-15T14:23:45.123456",
  "estimated_speed_mps": 15.6,
  "estimated_speed_mph": 34.9,
  "speed_limit_mph": 25.0,
  "position": [1.2, 3.4, 0.8],
  "frame_number": 1234,
  "image_path": "./violations/images/violation_42_1234.jpg",
  "metadata": {
    "total_detections": 15,
    "velocity_vector": [12.3, 8.9, 0.2]
  }
}
```

## Configuration

Configuration is managed through [config.py](src/speeding_detector/config.py) with the following parameters:

### Camera Configuration
- `width`, `height`: Frame dimensions (default: 640x480)
- `fps`: Frame rate (default: 30)
- `spatial_filter_sigma`: Spatial filter parameter (default: 2.0)
- `temporal_filter_alpha`: Temporal smoothing (default: 0.7)

### Detection Configuration
- `use_brisk`: Use BRISK vs SIFT features (default: True)
- `ground_plane_threshold`: Height threshold in meters (default: 0.5)
- `min_points_per_vehicle`: Minimum points for detection (default: 100)
- `max_matching_distance`: Maximum distance for matching (default: 2.0m)

### Tracking Configuration
- `process_noise`: Kalman filter process noise (default: 0.1)
- `measurement_noise`: Measurement noise (default: 0.5)
- `max_frames_missing`: Frames before track removal (default: 10)

### Speed Configuration
- `speed_limit_mph`: Speed limit (default: 25.0)
- `save_violation_images`: Save snapshots (default: True)
- `output_directory`: Output path (default: "./violations")

## Measurement of Success

As specified in the proposal, the system measures success over a 10-minute capture session:

1. **Total vehicles counted**: Number of unique vehicles detected
2. **Speeding violations**: Number of vehicles exceeding the speed limit
3. **Captured snapshots**: Images and metadata of offending vehicles
4. **Detection accuracy**: Ratio of detected to actual vehicles (requires ground truth)

## Technical Details

### Forward Kinematics

The transformation matrix for each joint uses the modified DH convention:

```
T_i = [cos(∏_i)  -sin(∏_i)cos(±_i)   sin(∏_i)sin(±_i)   a_i*cos(∏_i)]
      [sin(∏_i)   cos(∏_i)cos(±_i)  -cos(∏_i)sin(±_i)   a_i*sin(∏_i)]
      [0          sin(±_i)            cos(±_i)            d_i         ]
      [0          0                   0                   1           ]
```

End effector pose: TÄãÜ = TÅ ◊ TÇ ◊ TÉ ◊ TÑ ◊ TÖ ◊ TÜ

### Point Cloud Transformation

Each 3D point in camera frame is transformed to world frame:

```
p^world = TÄãÜ ◊ p^cam
```

### Speed Computation

Speed is computed from successive centroid positions:

```
v = ||P_{t+1} - P_t|| / ît
```

where:
- P_t is the centroid position at time t
- ît is the time between frames

### Kalman Filter State

State vector: [x, y, z, vx, vy, vz]
- Position: (x, y, z) in meters
- Velocity: (vx, vy, vz) in m/s

Uses constant velocity motion model with measurement updates from detected centroids.

## Limitations and Future Work

### Current Limitations
1. **Mock RealSense interface**: Requires actual hardware for deployment
2. **Simplified segmentation**: Uses grid-based clustering instead of neural networks
3. **Basic occlusion handling**: Could be improved with multi-hypothesis tracking
4. **Static camera assumption**: Algorithm assumes camera orientation remains relatively stable

### Future Enhancements
1. **Advanced segmentation**: Implement deep learning-based vehicle segmentation
2. **License plate recognition**: Extract plate numbers from violation images
3. **Path planning integration**: Automatically adjust robot arm to track vehicles
4. **Multi-camera fusion**: Use multiple RealSense cameras for better coverage
5. **Real-time visualization**: Add GUI for live monitoring

## Dependencies

Core dependencies managed by uv:
- `numpy` - Numerical operations and linear algebra
- `opencv-python` - BRISK/SIFT feature extraction and image processing
- `filterpy` - Kalman filtering implementation
- `scipy` - Scientific computing utilities

Hardware dependencies (deployment only):
- `pyrealsense2` - Intel RealSense SDK (x86_64 Linux only)
- `xarm-python-sdk` - UFACTORY Lite6 control (optional, for robot integration)

## Project Structure

```
cpsc5800-final-project/
   src/
      speeding_detector/
          __init__.py
          forward_kinematics.py    # DH parameters and FK solver
          camera_transform.py      # Point cloud transformation
          vehicle_detector.py      # Vehicle segmentation and features
          tracking.py              # Kalman filter tracking
          speed_estimator.py       # Speed computation and violations
          pipeline.py              # Main integration pipeline
          config.py                # Configuration parameters
   examples/
      run_detection.py             # Example usage script
   tests/                           # Unit tests (to be added)
   violations/                      # Output directory (generated)
      violations.json
      images/
   pyproject.toml                   # uv project configuration
   proposal.tex                     # Original project proposal
   README.md                        # This file
```

## Authors

Braeden Cullen - Yale University CPSC5800

## Acknowledgments

- UFACTORY for the Lite6 robotic arm platform
- Intel for the RealSense depth camera technology
- OpenCV community for computer vision tools
