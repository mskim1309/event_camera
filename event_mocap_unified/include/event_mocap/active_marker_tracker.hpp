// Copyright (c) 2025, Robot Vision Lab. SNU
// All rights reserved.
//
// This file uses Prophesee Metavision SDK.
// Reference: https://docs.prophesee.ai/

#pragma once

#ifdef HAS_METAVISION

#include <functional>
#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

#include "event_mocap/types.hpp"

namespace Metavision {
class Camera;
}

namespace emocap {

using TrackingCallback = std::function<void(
    Timestamp ts, const std::vector<MarkerObservation>& obs)>;

struct TrackerParams {
    // Modulated light detector
    int num_bits          = 8;
    std::uint32_t base_period_us = 400;
    float det_tolerance   = 0.3f;

    // Active marker tracker
    float tracker_radius  = 30.0f;
    float monitoring_freq_hz = 30.f;
    int64_t inactivity_period_us = 1000;
    float alpha_pos       = 0.05f;
    float distance_pct    = 0.3f;
    bool update_radius    = true;

    // Observation buffering
    size_t max_samples_per_id = 50;
    int64_t max_obs_age_us    = 2000;

    // Output trigger
    size_t delta_n_events = 5000;
};

class ActiveMarkerTracker {
public:
    ActiveMarkerTracker(const ActiveMarker& marker, const TrackerParams& params);
    ~ActiveMarkerTracker();

    void attach(Metavision::Camera& camera);
    void setCallback(TrackingCallback cb);
    void reset();

    using ObsBuffer = std::unordered_map<std::uint32_t,
        std::vector<std::pair<cv::Point2f, Timestamp>>>;
    const ObsBuffer& obsBuffer() const;

    TrackerParams& params();
    const TrackerParams& params() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace emocap

#endif  // HAS_METAVISION
