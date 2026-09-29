#include <thread>
#include <chrono>
#include <optional>
#include <fstream>
#include <set>
#include <vector>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/program_options.hpp>

#include <metavision/hal/facilities/i_events_stream_decoder.h>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/cv/events/event_source_id.h>
#include <metavision/sdk/cv/algorithms/modulated_light_detector_algorithm.h>
#include <metavision/sdk/cv/algorithms/active_marker_tracker_algorithm.h>
#include <metavision/sdk/driver/camera.h>

namespace detail {
    using ActiveMarker = std::unordered_map<std::uint32_t, cv::Point3f>;

    ActiveMarker load_active_marker(const std::string &path) {
        namespace pt = boost::property_tree;
        ActiveMarker active_marker;
        pt::ptree root;
        pt::read_json(path, root);
        const auto marker_node = root.get_child("active marker");
        std::vector<float> xyz;
        for (const auto &led_node : marker_node) {
            const auto id = led_node.second.get<std::uint32_t>("led.id");
            xyz.clear();
            const auto xyz_node = led_node.second.get_child("led.xyz");
            for (const auto &v : xyz_node) {
                xyz.emplace_back(v.second.get_value<float>());
            }
            active_marker[id] = cv::Point3f{xyz[0], xyz[1], xyz[2]};
        }
        return active_marker;
    }
} // namespace detail

struct Config {
    std::string event_file_path;
    std::string cam_config_path;
    std::string am_json_path;
    std::string calib_json_path;
    bool realtime_playback_speed;
    Metavision::ModulatedLightDetectorAlgorithm::Params detector_params;
    Metavision::ActiveMarkerTrackerAlgorithm::Params tracker_params;
    size_t delta_n_events;
    Metavision::timestamp delta_ts;
    float estimation_rate_hz;
};

std::optional<Config> parse_command_line(int argc, char *argv[]) {
    namespace po = boost::program_options;
    const std::string program_desc("Active marker 3D tracking without visualization");

    Config config;
    po::options_description options_desc;
    po::options_description base_options("Base options");
    base_options.add_options()
        ("help,h", "Produce help message.")
        ("input-event-file,i", po::value<std::string>(&config.event_file_path), "Path to input event file (RAW or HDF5). If not specified, the camera live stream is used.")
        ("input-camera-config,j", po::value<std::string>(&config.cam_config_path), "Path to a JSON file containing camera settings.")
        ("am-json-file,a", po::value<std::string>(&config.am_json_path)->required(), "Path to a JSON file describing an active marker.")
        ("calib-json-file,c", po::value<std::string>(&config.calib_json_path)->required(), "Path to a JSON file containing the camera calibration.");

    po::options_description detector_options("Modulated light detection options");
    int num_bits = 8;
    detector_options.add_options()
        ("detector-num-bits", po::value<int>(&num_bits)->default_value(8), "Number of bits encoding a light ID")
        ("detector-base-period-us", po::value<std::uint32_t>(&config.detector_params.base_period_us)->default_value(200), "Base period for modulated light decoding (us)")
        ("detector-tolerance", po::value<float>(&config.detector_params.tolerance)->default_value(0.1f), "Tolerance percentage on the blink period measurement");

    po::options_description tracker_options("Active marker tracker options");
    tracker_options.add_options()
        ("tracker-update-radius", po::value<bool>(&config.tracker_params.update_radius)->default_value(true))
        ("tracker-inactivity-period-us", po::value<Metavision::timestamp>(&config.tracker_params.inactivity_period_us)->default_value(1000))
        ("tracker-monitoring-frequency", po::value<float>(&config.tracker_params.monitoring_frequency_hz)->default_value(30.f))
        ("tracker-radius", po::value<float>(&config.tracker_params.radius)->default_value(30.f))
        ("tracker-distance-percentage", po::value<float>(&config.tracker_params.distance_pct)->default_value(0.3f))
        ("tracker-alpha-pos", po::value<float>(&config.tracker_params.alpha_pos)->default_value(0.05f));
    
    po::options_description pose_estimation_options("Pose estimation options");
    pose_estimation_options.add_options()
        ("pose-estimation-n-events", po::value<size_t>(&config.delta_n_events)->default_value(5000))
        ("pose-estimation-n-us", po::value<Metavision::timestamp>(&config.delta_ts)->default_value(0))
        ("pose-estimation-frequency-hz", po::value<float>(&config.estimation_rate_hz)->default_value(0.f));
        
    options_desc.add(base_options)
        .add(detector_options)
        .add(tracker_options)
        .add(pose_estimation_options);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(options_desc).run(), vm);
        po::notify(vm);
    } catch (po::error &e) {
        MV_LOG_ERROR() << program_desc;
        MV_LOG_ERROR() << options_desc;
        MV_LOG_ERROR() << "Parsing error:" << e.what();
        return std::nullopt;
    }

    config.detector_params.num_bits = static_cast<std::uint8_t>(num_bits);
    
    if (config.event_file_path.empty() && config.cam_config_path.empty()) {
        MV_LOG_ERROR() << "Camera config file is required for live execution";
        return std::nullopt;
    }

    if (vm.count("help")) {
        MV_LOG_INFO() << program_desc;
        MV_LOG_INFO() << options_desc;
        return std::nullopt;
    }

    return config;
}

int main(int argc, char *argv[]) {
    auto opt_config = parse_command_line(argc, argv);
    if (!opt_config) {
        return 1;
    }

    Metavision::Camera camera;
    if (opt_config->event_file_path.empty()) {
        Metavision::DeviceConfig device_config;
        device_config.enable_biases_range_check_bypass(true);
        camera = Metavision::Camera::from_first_available(device_config);
        if (!opt_config->cam_config_path.empty()) {
            camera.load(opt_config->cam_config_path);
        }
    } else {
        const auto cam_config = 
            Metavision::FileConfigHints().real_time_playback(
                opt_config->realtime_playback_speed);
        camera = 
            Metavision::Camera::from_file(opt_config->event_file_path, cam_config);
    }

    // Load 3D active marker definition (LED IDs and 3D positions)
    const auto active_marker = detail::load_active_marker(opt_config->am_json_path);

    // Setup detector dimensions
    const auto eb_w = static_cast<std::uint16_t>(camera.geometry().width());
    const auto eb_h = static_cast<std::uint16_t>(camera.geometry().height());
    opt_config->detector_params.width  = eb_w;
    opt_config->detector_params.height = eb_h;

    opt_config->detector_params.tolerance = 0.3f;
    opt_config->detector_params.base_period_us = 400;  

    // Initialize modulated light detector
    Metavision::ModulatedLightDetectorAlgorithm modulated_light_detector(
        opt_config->detector_params);

    // Prepare the tracker with the set of LED IDs
    std::set<std::uint32_t> led_ids;
    for (const auto &kv : active_marker) {
        led_ids.insert(kv.first);
    }

    Metavision::ActiveMarkerTrackerAlgorithm tracker(opt_config->tracker_params, led_ids);

    // Load camera intrinsics from calibration JSON
    cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat dist_coeffs;
    try {
        namespace pt = boost::property_tree;
        pt::ptree root;
        pt::read_json(opt_config->calib_json_path, root);
        pt::ptree intrinsics;
        if (auto intr = root.get_child_optional("intrinsics")) {
            intrinsics = intr.get();
        } else if (auto proj = root.get_child_optional("proj_master")) {
            intrinsics = proj.get();
        } else {
            intrinsics = root;
        }
        std::vector<double> K, D;
        if (auto knode = intrinsics.get_child_optional("K")) {
            for (auto &v : intrinsics.get_child("K")) {
                K.push_back(v.second.get_value<double>());
            }
        }
        if (auto dnode = intrinsics.get_child_optional("D")) {
            for (auto &v : intrinsics.get_child("D")) {
                D.push_back(v.second.get_value<double>());
            }
        }
        if (K.size() == 9) {
            camera_matrix.at<double>(0, 0) = K[0];
            camera_matrix.at<double>(0, 2) = K[2];
            camera_matrix.at<double>(1, 1) = K[4];
            camera_matrix.at<double>(1, 2) = K[5];
        }
        if (!D.empty()) {
            dist_coeffs = cv::Mat::zeros(1, (int)D.size(), CV_64F);
            for (int i = 0; i < (int)D.size(); ++i) {
                dist_coeffs.at<double>(0, i) = D[i];
            }
        }
    } catch (const std::exception &e) {
        MV_LOG_ERROR() << "Failed to load calibration: " << e.what();
    }

    // Prepare vectors and counters for the event loop
    std::vector<Metavision::EventSourceId> source_id_events;
    std::vector<Metavision::EventActiveTrack> active_tracks;
    size_t event_counter = 0;
    Metavision::timestamp last_output_ts = 0;
    size_t event_threshold = opt_config->delta_n_events;
    Metavision::timestamp time_threshold = opt_config->delta_ts;
    if (opt_config->estimation_rate_hz > 0) {
        time_threshold = static_cast<Metavision::timestamp>(
            1000000.0f / opt_config->estimation_rate_hz);
        event_threshold = 0;
    }

    // Open output CSV file for poses
    std::ofstream pose_file("pose_log.csv");
    pose_file << "timestamp,x,y,z\n";

    // Time callback for tracker maintenance
    auto decoder = camera.get_device().get_facility<Metavision::I_EventsStreamDecoder>();
    decoder->add_time_callback([&](Metavision::timestamp t) {
        tracker.notify_elapsed_time(t);
    });

    // Event callback: detect LED events, track LEDs, compute pose via solvePnP
    camera.cd().add_callback([&](const auto begin, const auto end) {
        source_id_events.clear();
        modulated_light_detector.process_events(
            begin, end, std::back_inserter(source_id_events));

        active_tracks.clear();
        tracker.process_events(
            source_id_events.cbegin(), source_id_events.cend(),
            std::back_inserter(active_tracks));

        // Count events for scheduling
        size_t new_events = std::distance(
            source_id_events.cbegin(), source_id_events.cend());
        event_counter += new_events;

        // Determine current timestamp from tracks (or last event)
        Metavision::timestamp current_ts = 0;
        for (const auto &track : active_tracks) {
            current_ts = std::max(current_ts, track.t);
        }
        if (current_ts == 0 && !source_id_events.empty()) {
            current_ts = source_id_events.back().t;
        }

        bool time_ok = (time_threshold > 0 && current_ts - last_output_ts >= time_threshold);
        bool count_ok = (event_threshold > 0 && event_counter >= event_threshold);
        if ((time_threshold == 0 && event_threshold == 0) || time_ok || count_ok) {
            // Build 2D-3D correspondences
            std::vector<cv::Point3f> pts3d;
            std::vector<cv::Point2f> pts2d;
            for (const auto &track : active_tracks) {
                auto it = active_marker.find(track.id);
                if (it != active_marker.end()) {
                    pts3d.push_back(it->second);
                    pts2d.emplace_back(track.x, track.y);
                }
            }
            if (pts3d.size() >= 4) {
                cv::Mat rvec, tvec;
                bool success = cv::solvePnP(pts3d, pts2d, camera_matrix, 
                                           dist_coeffs, rvec, tvec);
                if (success) {
                    double tx = tvec.at<double>(0);
                    double ty = tvec.at<double>(1);
                    double tz = tvec.at<double>(2);
                    pose_file << current_ts << "," << tx << "," << ty << "," << tz << "\n";
                }
                // Reset after solving
                last_output_ts = current_ts;
                event_counter = 0;
            } else {
                // Not enough points: reset counter but keep last_output_ts
                event_counter = 0;
            }
        }
    });

    camera.start();
    while (camera.is_running()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    camera.stop();

    return 0;
}
