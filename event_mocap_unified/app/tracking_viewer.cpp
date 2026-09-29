// Copyright (c) 2025, Robot Vision Lab. SNU
// All rights reserved.
//
// 2D active marker tracking viewer.
// Uses Prophesee Metavision SDK for event processing and visualization.
// Reference: https://docs.prophesee.ai/

#ifdef HAS_METAVISION

#include <iostream>
#include <thread>
#include <set>
#include <unordered_map>
#include <vector>
#include <cstdlib>

#include <opencv2/opencv.hpp>
#include <boost/circular_buffer.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <metavision/hal/facilities/i_events_stream_decoder.h>
#include <metavision/sdk/cv/events/event_source_id.h>
#include <metavision/sdk/cv/algorithms/modulated_light_detector_algorithm.h>
#include <metavision/sdk/cv/algorithms/active_marker_tracker_algorithm.h>
#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/ui/utils/mt_window.h>
#include <metavision/sdk/ui/utils/event_loop.h>

#include "util/yaml.h"
#include "event_mocap/gflags.h"
#include "event_mocap/config_loader.hpp"

// --- Track trail renderer ---
class TrackPrinter {
public:
    TrackPrinter(int width, int height, const std::set<std::uint32_t>& ids,
                 std::uint32_t trail_length)
        : width_(width), height_(height)
    {
        for (auto id : ids) {
            tracks_[id] = boost::circular_buffer<cv::Point2f>(trail_length);
            updated_[id] = false;
        }
        // Generate random colors per ID
        for (size_t i = 0; i < 500; ++i)
            colors_.push_back(cv::Vec3b(rand() % 255, rand() % 255, rand() % 255));
    }

    template <typename It>
    void process(It begin, It end) {
        for (auto it = begin; it != end; ++it) {
            if (it->status == Metavision::EventActiveTrack::Status::Lost) continue;
            tracks_[it->id].push_back(cv::Point2f(it->x, it->y));
            radius_[it->id] = it->radius;
            updated_[it->id] = true;
        }
    }

    void draw(cv::Mat& frame) {
        // Fade out non-updated tracks
        for (auto& [id, upd] : updated_) {
            if (!upd) {
                auto& buf = tracks_[id];
                size_t n = std::min<size_t>(100, buf.size());
                buf.erase_begin(n);
            }
        }

        // Draw trails + labels
        for (const auto& [id, buf] : tracks_) {
            if (buf.empty()) continue;
            auto color = colors_[id % colors_.size()];

            for (const auto& pt : buf)
                frame.at<cv::Vec3b>(
                    std::clamp<int>(pt.y, 0, height_ - 1),
                    std::clamp<int>(pt.x, 0, width_ - 1)) = color;

            // LED ID text
            cv::putText(frame, std::to_string(id), buf.back(),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(color[0], color[1], color[2]), 1);

            // Radius circle
            if (radius_.count(id))
                cv::circle(frame, buf.back(), static_cast<int>(radius_[id]),
                           cv::Scalar(color[0], color[1], color[2]));
        }

        // Reset update flags
        for (auto& [id, upd] : updated_) upd = false;
    }

private:
    int width_, height_;
    std::unordered_map<std::uint32_t, boost::circular_buffer<cv::Point2f>> tracks_;
    std::unordered_map<std::uint32_t, bool> updated_;
    std::unordered_map<std::uint32_t, float> radius_;
    std::vector<cv::Vec3b> colors_;
};

// Additional flags for this viewer
DEFINE_int32(trail_length, 5000, "Max displayed track trail length");
DEFINE_bool(show_events, true, "Show accumulated event image as background");

int main(int argc, char* argv[]) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);

    if (!FLAGS_config.empty())
        rvl::LoadYAMLConfigAndSyncGoogleFlags(FLAGS_config);

    if (FLAGS_marker_json.empty()) {
        LOG(ERROR) << "-marker_json is required";
        return 1;
    }

    // Load marker IDs
    auto marker = emocap::loadActiveMarker(FLAGS_marker_json);
    std::set<std::uint32_t> led_ids;
    for (const auto& [id, _] : marker) led_ids.insert(id);

    // Setup camera
    Metavision::Camera camera;
    if (!FLAGS_input.empty()) {
        auto hints = Metavision::FileConfigHints().real_time_playback(FLAGS_realtime);
        camera = Metavision::Camera::from_file(FLAGS_input, hints);
        LOG(INFO) << "Playing file: " << FLAGS_input;
    } else {
        Metavision::DeviceConfig dev_cfg;
        dev_cfg.enable_biases_range_check_bypass(true);
        camera = Metavision::Camera::from_first_available(dev_cfg);
        if (!FLAGS_camera_config.empty()) camera.load(FLAGS_camera_config);
        LOG(INFO) << "Live camera";
    }

    int w = camera.geometry().width();
    int h = camera.geometry().height();

    // SDK algorithms
    Metavision::ModulatedLightDetectorAlgorithm::Params det_params;
    det_params.num_bits       = static_cast<std::uint8_t>(FLAGS_det_num_bits);
    det_params.base_period_us = static_cast<std::uint32_t>(FLAGS_det_base_period_us);
    det_params.tolerance      = static_cast<float>(FLAGS_det_tolerance);
    det_params.width          = static_cast<std::uint16_t>(w);
    det_params.height         = static_cast<std::uint16_t>(h);

    Metavision::ModulatedLightDetectorAlgorithm detector(det_params);

    Metavision::ActiveMarkerTrackerAlgorithm::Params trk_params;
    trk_params.radius                = static_cast<float>(FLAGS_trk_radius);
    trk_params.monitoring_frequency_hz = static_cast<float>(FLAGS_trk_monitoring_freq_hz);
    trk_params.inactivity_period_us  = FLAGS_trk_inactivity_period_us;
    trk_params.alpha_pos             = static_cast<float>(FLAGS_trk_alpha_pos);
    trk_params.distance_pct          = static_cast<float>(FLAGS_trk_distance_pct);
    trk_params.update_radius         = true;

    Metavision::ActiveMarkerTrackerAlgorithm tracker(trk_params, led_ids);

    // Display
    Metavision::MTWindow window("Active Marker Tracking", w, h,
                                 Metavision::BaseWindow::RenderMode::BGR);
    window.set_keyboard_callback([&](auto key, int, auto action, int) {
        if (action == Metavision::UIAction::RELEASE &&
            (key == Metavision::UIKeyEvent::KEY_ESCAPE || key == Metavision::UIKeyEvent::KEY_Q))
            window.set_close_flag();
    });

    TrackPrinter printer(w, h, led_ids, FLAGS_trail_length);
    cv::Mat frame(h, w, CV_8UC3, cv::Scalar(0));
    cv::Mat event_frame(h, w, CV_8UC3, cv::Scalar(0));  // accumulated event image

    // Slicer for display refresh (~50fps)
    Metavision::EventBufferReslicerAlgorithm slicer;
    slicer.set_slicing_condition(
        Metavision::EventBufferReslicerAlgorithm::Condition::make_n_us(20000));
    slicer.set_on_new_slice_callback(
        [&](auto, Metavision::timestamp, std::size_t) {
            if (FLAGS_show_events)
                event_frame.copyTo(frame);
            else
                frame.setTo(0);
            printer.draw(frame);
            window.show_async(frame);
            event_frame.setTo(0);
        });

    // Buffers
    std::vector<Metavision::EventSourceId> src_events;
    std::vector<Metavision::EventActiveTrack> active_tracks;

    // Time callback
    auto decoder = camera.get_device().get_facility<Metavision::I_EventsStreamDecoder>();
    if (decoder) {
        decoder->add_time_callback([&](Metavision::timestamp t) {
            tracker.notify_elapsed_time(t);
            slicer.notify_elapsed_time(t);
        });
    }

    // Event callback: accumulate events into background + run tracking
    camera.cd().add_callback([&](const auto begin, const auto end) {
        // Accumulate raw CD events into event_frame
        for (auto it = begin; it != end; ++it) {
            int x = it->x, y = it->y;
            if (x >= 0 && x < w && y >= 0 && y < h) {
                if (it->p == 1)  // ON event — blue-white
                    event_frame.at<cv::Vec3b>(y, x) = cv::Vec3b(200, 180, 150);
                else             // OFF event — dark red
                    event_frame.at<cv::Vec3b>(y, x) = cv::Vec3b(50, 50, 150);
            }
        }

        // Modulated light detection + tracking
        src_events.clear();
        detector.process_events(begin, end, std::back_inserter(src_events));

        active_tracks.clear();
        tracker.process_events(src_events.cbegin(), src_events.cend(),
                               std::back_inserter(active_tracks));

        slicer.process_events(active_tracks.cbegin(), active_tracks.cend(),
            [&](const auto sb, const auto se) { printer.process(sb, se); });
    });

    // Run
    camera.start();
    while (camera.is_running()) {
        if (window.should_close()) break;
        Metavision::EventLoop::poll_and_dispatch(20);
    }
    camera.stop();

    return 0;
}

#else  // !HAS_METAVISION

#include <iostream>
int main() {
    std::cerr << "tracking_viewer requires MetavisionSDK. Build with MetavisionSDK to enable." << std::endl;
    return 1;
}

#endif  // HAS_METAVISION
