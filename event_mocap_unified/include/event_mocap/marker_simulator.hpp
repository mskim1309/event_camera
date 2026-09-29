// Copyright (c) 2025, Robot Vision Lab. SNU
// All rights reserved.

#pragma once

#include <functional>
#include <vector>
#include <random>
#include "event_mocap/types.hpp"

namespace emocap {

enum class TrajectoryType { CIRCULAR, LINEAR, SINUSOIDAL };

struct SimConfig {
    TrajectoryType traj_type = TrajectoryType::CIRCULAR;
    double duration_s   = 10.0;    // total simulation time (seconds)
    double rate_hz      = 100.0;   // observation rate
    double pixel_noise  = 0.5;     // gaussian noise on 2D observations (pixels)

    // Circular trajectory params
    double circle_radius  = 200.0;  // mm
    double circle_height  = 500.0;  // mm (distance from marker plane)
    double circle_freq    = 0.5;    // Hz

    // Linear trajectory params
    double linear_speed   = 50.0;   // mm/s along x
    double linear_z       = 500.0;  // mm

    // Sinusoidal trajectory params
    double sin_amp_x = 100.0, sin_amp_y = 50.0, sin_amp_z = 30.0;
    double sin_freq_x = 0.3, sin_freq_y = 0.5, sin_freq_z = 0.2;
    double sin_base_z = 500.0;
};

// Callbacks
using ObservationCallback = std::function<void(
    Timestamp ts, const std::vector<MarkerObservation>& obs)>;
using GroundTruthCallback = std::function<void(
    Timestamp ts, const Pose& gt_pose)>;

class MarkerSimulator {
public:
    MarkerSimulator(const ActiveMarker& marker,
                    const CameraIntrinsics& intrinsics,
                    const SimConfig& config);

    // Run the full simulation, invoking callbacks at each timestep
    void run(ObservationCallback obs_cb,
             GroundTruthCallback gt_cb = nullptr);

private:
    // Compute camera pose (world→camera) at time t
    Pose cameraPoseAt(double t_sec) const;

    // Project marker 3D points to 2D given pose
    std::vector<MarkerObservation> projectMarkers(const Pose& pose) const;

    ActiveMarker marker_;
    CameraIntrinsics intrinsics_;
    SimConfig config_;
    mutable std::mt19937 rng_{42};
};

}  // namespace emocap
