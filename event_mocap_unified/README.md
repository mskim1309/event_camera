# Event MoCap Unified

Active marker motion capture system based on event cameras.
Uses Prophesee event cameras with modulated LED markers for real-time 6-DOF pose estimation.

## Pipeline

```
Data Source (sim / file / live)
    |
    v
2D Marker Observations (LED ID + pixel coordinates)
    |
    v
PnP + Sliding-Window Bundle Adjustment (Ceres)
    |
    v
6-DOF Camera Pose --> CSV output + Rerun visualization
```

## Build

### Dependencies

| Library | Purpose | Required |
|---------|---------|----------|
| Eigen3, OpenCV, Ceres, glog, gflags, yaml-cpp | Core | Yes |
| MetavisionSDK | Event camera 2D tracking | Optional (file/live) |
| rerun_sdk | 3D visualization | Optional |

### Instructions

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

## Execution

All scripts run from the project root. Defaults are loaded from `config/default.yaml`.
Any flag can be overridden via CLI: `./run_sim.sh -sim_duration=20 -sim_pixel_noise=2.0`

### Simulation (no MetavisionSDK needed)
```bash
./run_sim.sh
```
Generates a virtual camera trajectory, projects markers to 2D with Gaussian noise,
then recovers the pose via PnP+BA. Trajectory types: `circular`, `linear`, `sinusoidal`.

### Recorded Data (requires MetavisionSDK)
```bash
./run_file_0819.sh
```
Plays back `.raw` event files through the full pipeline.

### 2D Tracking Viewer (requires MetavisionSDK)
```bash
./run_tracking_viewer.sh
```
SDK-native window showing event image + 2D marker trails + LED IDs.
Supports both recorded files and live camera (`-show_events=false` to hide event background).

### Live Camera (requires MetavisionSDK)
```bash
./build/event_mocap_unified -config=config/default.yaml -mode=live -visualize
```

## Output

CSV saved to `result/` (default: `result/pose_log.csv`).

```
timestamp,x,y,z,rvec_x,rvec_y,rvec_z
```
- `x, y, z`: camera center in world frame (mm)
- `rvec`: angle-axis rotation (world-to-camera)

```bash
python3 scripts/pose_visualization.py result/pose_log.csv
```

## Project Structure

```
event_mocap_unified/
├── app/
│   ├── main.cpp                        # Main entry point
│   └── tracking_viewer.cpp             # [SDK] 2D tracking viewer
├── include/event_mocap/
│   ├── types.hpp                       # Common types (Pose, ActiveMarker, ...)
│   ├── gflags.h                        # Parameter declarations
│   ├── config_loader.hpp               # JSON loading (markers, calibration)
│   ├── pose_estimator.hpp              # PnP + sliding-window BA
│   ├── marker_simulator.hpp            # Simulation trajectory + 2D projection
│   ├── active_marker_tracker.hpp       # [SDK] 2D tracking wrapper
│   ├── data_source.hpp                 # DataSource interface (sim/file/live)
│   └── visualizer.hpp                  # Rerun visualization
├── src/
│   ├── pose_estimator.cpp              # PnP + Ceres BA (core algorithm)
│   ├── marker_simulator.cpp            # Trajectory + projectPoints + noise
│   ├── active_marker_tracker.cpp       # [SDK] ModulatedLightDetector + Tracker
│   ├── data_source.cpp                 # Sim / EventFile / LiveCamera sources
│   ├── visualizer.cpp                  # Rerun logging (conditional)
│   ├── config_loader.cpp               # JSON parser
│   ├── gflags.cpp                      # Parameter defaults
│   └── util/                           # RVL shared utilities
├── config/
│   ├── default.yaml                    # Default configuration
│   ├── calibration_*.json              # Camera intrinsic calibration
│   ├── camera_config.json              # Camera bias settings (live only)
│   └── marker/                         # Marker 3D coordinates + LED IDs
├── result/                             # CSV output
├── scripts/                            # Python visualization
├── run_sim.sh                          # Run simulation
├── run_file_0819.sh                    # Run with 0819 dataset
├── run_tracking_viewer.sh              # Run 2D tracking viewer
└── CMakeLists.txt
```

## Key Parameters

All parameters are configurable via `config/default.yaml` or CLI flags.
Full list: `./build/event_mocap_unified -help`

| Flag | Default | Description |
|------|---------|-------------|
| **Pose Estimation** | | |
| `-ba_window_size` | 5 | BA sliding window frame count |
| **2D Tracking** | | |
| `-det_base_period_us` | 400 | LED blink base period (us) |
| `-trk_radius` | 30.0 | Tracker search radius (px) |
| `-obs_delta_n_events` | 5000 | Pose estimation trigger interval |
| **Simulation** | | |
| `-trajectory` | circular | circular / linear / sinusoidal |
| `-sim_duration` | 10.0 | Duration (s) |
| `-sim_rate_hz` | 100.0 | Observation rate (Hz) |
| `-sim_pixel_noise` | 0.5 | 2D Gaussian noise std (px) |

## References

- [Metavision Active Marker 2D Tracking (SDK docs)](https://docs.prophesee.ai/4.6.2/samples/modules/cv/active_marker_2d_tracking_cpp.html)
