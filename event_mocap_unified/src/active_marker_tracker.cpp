// Copyright (c) 2025, Robot Vision Lab. SNU
// All rights reserved.
//
// This file uses Prophesee Metavision SDK.
// Reference: https://docs.prophesee.ai/

#ifdef HAS_METAVISION

#include "event_mocap/active_marker_tracker.hpp"

#include <algorithm>
#include <iostream>
#include <set>

#include <metavision/sdk/cv/algorithms/modulated_light_detector_algorithm.h>
#include <metavision/sdk/cv/algorithms/active_marker_tracker_algorithm.h>
#include <metavision/sdk/driver/camera.h>
#include <metavision/hal/facilities/i_events_stream_decoder.h>

namespace emocap {

struct ActiveMarkerTracker::Impl {
    ActiveMarker marker;
    TrackerParams params;
    TrackingCallback callback;

    // SDK algorithm instances
    std::shared_ptr<Metavision::ModulatedLightDetectorAlgorithm> detector;
    std::shared_ptr<Metavision::ActiveMarkerTrackerAlgorithm> tracker;

    // Internal buffers
    std::vector<Metavision::EventSourceId> source_id_events;
    std::vector<Metavision::EventActiveTrack> active_tracks;
    ObsBuffer obs_buffer;
    size_t event_counter = 0;
    Timestamp last_output_ts = 0;

    // ---------------------------------------------------------------
    // Initialize SDK algorithms
    // ---------------------------------------------------------------
    void initAlgorithms(int width, int height) {
        Metavision::ModulatedLightDetectorAlgorithm::Params det_params;
        det_params.num_bits       = static_cast<std::uint8_t>(params.num_bits);
        det_params.base_period_us = params.base_period_us;
        det_params.tolerance      = params.det_tolerance;
        det_params.width          = static_cast<std::uint16_t>(width);
        det_params.height         = static_cast<std::uint16_t>(height);
        detector = std::make_shared<Metavision::ModulatedLightDetectorAlgorithm>(det_params);

        std::set<std::uint32_t> led_ids;
        for (const auto& [id, _] : marker) led_ids.insert(id);

        Metavision::ActiveMarkerTrackerAlgorithm::Params trk_params;
        trk_params.radius                = params.tracker_radius;
        trk_params.monitoring_frequency_hz = params.monitoring_freq_hz;
        trk_params.inactivity_period_us  = params.inactivity_period_us;
        trk_params.alpha_pos             = params.alpha_pos;
        trk_params.distance_pct          = params.distance_pct;
        trk_params.update_radius         = params.update_radius;
        tracker = std::make_shared<Metavision::ActiveMarkerTrackerAlgorithm>(trk_params, led_ids);
    }

    // ---------------------------------------------------------------
    // Process a batch of CD events (called by Camera callback)
    // ---------------------------------------------------------------
    template <typename It>
    void processEvents(It begin, It end) {
        // --- Step 1: Modulated light detection (decode LED IDs) ---
        source_id_events.clear();
        detector->process_events(begin, end, std::back_inserter(source_id_events));

        // --- Step 2: Active marker tracking (smooth 2D positions) ---
        active_tracks.clear();
        tracker->process_events(source_id_events.cbegin(), source_id_events.cend(),
                                std::back_inserter(active_tracks));

        event_counter += source_id_events.size();

        // --- Step 3: Determine current timestamp ---
        Timestamp current_ts = 0;
        for (const auto& t : active_tracks) {
            auto ts = static_cast<Timestamp>(t.t);
            if (ts > current_ts) current_ts = ts;
        }
        if (current_ts == 0 && !source_id_events.empty())
            current_ts = static_cast<Timestamp>(source_id_events.back().t);
        if (current_ts == 0) return;

        // --- Step 4: Update observation buffer ---
        updateObsBuffer(current_ts);

        // --- Step 5: Check trigger condition ---
        if (params.delta_n_events > 0 && event_counter < params.delta_n_events)
            return;

        // --- Step 6: Build averaged observations and emit ---
        auto observations = buildObservations(current_ts);
        if (callback && !observations.empty())
            callback(current_ts, observations);

        event_counter = 0;
        last_output_ts = current_ts;
    }

    // ---------------------------------------------------------------
    // Update observation buffer with latest tracks
    // ---------------------------------------------------------------
    void updateObsBuffer(Timestamp current_ts) {
        for (const auto& track : active_tracks) {
            auto& vec = obs_buffer[track.id];
            vec.emplace_back(cv::Point2f(track.x, track.y),
                             static_cast<Timestamp>(track.t));

            // Cap buffer size
            if (vec.size() > params.max_samples_per_id) {
                vec.erase(vec.begin(),
                          vec.begin() + static_cast<long>(vec.size() - params.max_samples_per_id));
            }
        }

        // Prune old observations
        auto max_age = params.max_obs_age_us;
        for (auto it = obs_buffer.begin(); it != obs_buffer.end();) {
            auto& vec = it->second;
            vec.erase(std::remove_if(vec.begin(), vec.end(),
                [&](const auto& p) {
                    return (current_ts > p.second) && (current_ts - p.second > max_age);
                }), vec.end());
            if (vec.empty()) it = obs_buffer.erase(it); else ++it;
        }
    }

    // ---------------------------------------------------------------
    // Build time-weighted average 2D observations
    // Modify this to experiment with different averaging / filtering.
    // ---------------------------------------------------------------
    std::vector<MarkerObservation> buildObservations(Timestamp current_ts) {
        std::vector<MarkerObservation> observations;

        for (const auto& [id, _] : marker) {
            auto buf_it = obs_buffer.find(id);
            if (buf_it == obs_buffer.end()) continue;
            const auto& samples = buf_it->second;
            if (samples.empty()) continue;

            // Time-weighted average: more recent samples get higher weight
            double sw = 0, sx = 0, sy = 0;
            for (const auto& [pt, ts] : samples) {
                auto dt = (current_ts > ts) ? (current_ts - ts) : 0;
                double w = 1.0 / (1.0 + static_cast<double>(dt));
                sx += pt.x * w;
                sy += pt.y * w;
                sw += w;
            }
            if (sw <= 0) continue;

            observations.push_back({id, cv::Point2f(
                static_cast<float>(sx / sw), static_cast<float>(sy / sw))});
        }

        return observations;
    }
};

// ============================================================
// Public API
// ============================================================

ActiveMarkerTracker::ActiveMarkerTracker(const ActiveMarker& marker,
                                         const TrackerParams& params)
    : impl_(std::make_unique<Impl>())
{
    impl_->marker = marker;
    impl_->params = params;
}

ActiveMarkerTracker::~ActiveMarkerTracker() = default;

void ActiveMarkerTracker::attach(Metavision::Camera& camera) {
    int w = camera.geometry().width();
    int h = camera.geometry().height();
    impl_->initAlgorithms(w, h);

    // Time callback for tracker temporal logic
    auto decoder = camera.get_device().get_facility<Metavision::I_EventsStreamDecoder>();
    if (decoder) {
        auto tracker = impl_->tracker;
        decoder->add_time_callback([tracker](Timestamp t) {
            tracker->notify_elapsed_time(t);
        });
    }

    // Event callback — main processing entry point
    camera.cd().add_callback(
        [this](const auto begin, const auto end) {
            impl_->processEvents(begin, end);
        });
}

void ActiveMarkerTracker::setCallback(TrackingCallback cb) {
    impl_->callback = cb;
}

void ActiveMarkerTracker::reset() {
    impl_->obs_buffer.clear();
    impl_->event_counter = 0;
    impl_->last_output_ts = 0;
}

const ActiveMarkerTracker::ObsBuffer& ActiveMarkerTracker::obsBuffer() const {
    return impl_->obs_buffer;
}

TrackerParams& ActiveMarkerTracker::params() { return impl_->params; }
const TrackerParams& ActiveMarkerTracker::params() const { return impl_->params; }

}  // namespace emocap

#endif  // HAS_METAVISION
