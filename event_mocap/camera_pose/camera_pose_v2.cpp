#include <thread>
#include <chrono>
#include <optional>
#include <fstream>
#include <set>
#include <vector>
#include <iostream>
#include <mutex>
#include <atomic>
#include <unordered_map>

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

// ======================
// Global pose variables
// ======================
std::mutex pose_mutex;
cv::Mat global_rvec = cv::Mat::zeros(3, 1, CV_64F);
cv::Mat global_tvec = cv::Mat::zeros(3, 1, CV_64F);
Metavision::timestamp global_timestamp = 0;
std::atomic<bool> new_pose_available{false};

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
}

// ======================
// Config parsing struct
// ======================
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
    const std::string program_desc("Active marker 3D tracking (CSV logging only)");

    Config config;
    po::options_description options_desc;
    po::options_description base_options("Base options");
    base_options.add_options()
        ("help,h", "Produce help message.")
        ("input-event-file,i", po::value<std::string>(&config.event_file_path), "Path to input event file (RAW or HDF5).")
        ("input-camera-config,j", po::value<std::string>(&config.cam_config_path), "Path to JSON camera settings.")
        ("am-json-file,a", po::value<std::string>(&config.am_json_path)->required(), "Path to active marker JSON file.")
        ("calib-json-file,c", po::value<std::string>(&config.calib_json_path)->required(), "Path to calibration JSON file.");

    po::options_description detector_options("Modulated light detection options");
    int num_bits = 8;
    detector_options.add_options()
        ("detector-num-bits", po::value<int>(&num_bits)->default_value(8))
        ("detector-base-period-us", po::value<std::uint32_t>(&config.detector_params.base_period_us)->default_value(200))
        ("detector-tolerance", po::value<float>(&config.detector_params.tolerance)->default_value(0.1f));

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

    options_desc.add(base_options).add(detector_options).add(tracker_options).add(pose_estimation_options);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(options_desc).run(), vm);
        if (vm.count("help")) {
            MV_LOG_INFO() << program_desc;
            MV_LOG_INFO() << options_desc;
            return std::nullopt;
        }
        po::notify(vm);
    } catch (po::error &e) {
        MV_LOG_ERROR() << program_desc;
        MV_LOG_ERROR() << options_desc;
        MV_LOG_ERROR() << "Parsing error:" << e.what();
        return std::nullopt;
    }

    config.detector_params.num_bits = static_cast<std::uint8_t>(num_bits);
    return config;
}

// =========================================
// PnP 이전 결과 저장용 변수
// =========================================
static bool pnp_initialized = false;
static cv::Mat rvec_prev, tvec_prev;

// =========================================
// solvePnP with extrinsic guess (군집 문제 해결)
// =========================================
bool solvePnP_stable(
        const std::vector<cv::Point3f>& pts3d,
        const std::vector<cv::Point2f>& pts2d,
        const cv::Mat& K,
        const cv::Mat& dist,
        cv::Mat& rvec,
        cv::Mat& tvec)
{
    if (!pnp_initialized) {
        // 첫 프레임: guess 없이 solvePnP
        bool ok = cv::solvePnP(
            pts3d, pts2d, K, dist,
            rvec, tvec,
            false,                   // ⭐ 첫 프레임은 guess 사용 X
            cv::SOLVEPNP_ITERATIVE
        );

        if (ok) {
            rvec_prev = rvec.clone();
            tvec_prev = tvec.clone();
            pnp_initialized = true;
        }
        return ok;
    }

    // 이후 프레임: 이전 rvec/tvec을 초기값으로 제공
    rvec = rvec_prev.clone();
    tvec = tvec_prev.clone();

    bool ok = cv::solvePnP(
        pts3d, pts2d, K, dist,
        rvec, tvec,
        true,                       // ⭐ 이전 해를 초기값으로 사용!
        cv::SOLVEPNP_ITERATIVE
    );

    if (ok) {
        rvec_prev = rvec.clone();
        tvec_prev = tvec.clone();
    }

    return ok;
}


int main(int argc, char *argv[]) {
    auto opt_config = parse_command_line(argc, argv);
    if (!opt_config) return 1;

    // CSV log file
    std::ofstream csv_file("camera_pose.csv");
    csv_file << "timestamp_us,x,y,z,"
             << "R00,R01,R02,R10,R11,R12,R20,R21,R22\n";

    // Camera setup
    Metavision::Camera camera;
    if (opt_config->event_file_path.empty()) {
        Metavision::DeviceConfig device_config;
        camera = Metavision::Camera::from_first_available(device_config);
        if (!opt_config->cam_config_path.empty())
            camera.load(opt_config->cam_config_path);
    } else {
        const auto cam_config = Metavision::FileConfigHints().real_time_playback(opt_config->realtime_playback_speed);
        camera = Metavision::Camera::from_file(opt_config->event_file_path, cam_config);
    }

    const auto active_marker = detail::load_active_marker(opt_config->am_json_path);

    // Detector setup
    const auto eb_w = static_cast<std::uint16_t>(camera.geometry().width());
    const auto eb_h = static_cast<std::uint16_t>(camera.geometry().height());
    opt_config->detector_params.width  = eb_w;
    opt_config->detector_params.height = eb_h;
    opt_config->detector_params.tolerance = 0.3f;
    opt_config->detector_params.base_period_us = 400;

    Metavision::ModulatedLightDetectorAlgorithm modulated_light_detector(opt_config->detector_params);

    std::set<std::uint32_t> led_ids;
    for (const auto &kv : active_marker) led_ids.insert(kv.first);
    Metavision::ActiveMarkerTrackerAlgorithm tracker(opt_config->tracker_params, led_ids);

    // Camera intrinsics
    cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat dist_coeffs;
    try {
        namespace pt = boost::property_tree;
        pt::ptree root;
        pt::read_json(opt_config->calib_json_path, root);
        auto intrinsics = root.get_child_optional("intrinsics").value_or(root);
        std::vector<double> K, D;
        for (auto &v : intrinsics.get_child("K")) K.push_back(v.second.get_value<double>());
        for (auto &v : intrinsics.get_child("D")) D.push_back(v.second.get_value<double>());
        if (K.size() == 9) {
            camera_matrix.at<double>(0, 0) = K[0];
            camera_matrix.at<double>(0, 2) = K[2];
            camera_matrix.at<double>(1, 1) = K[4];
            camera_matrix.at<double>(1, 2) = K[5];
        }
        if (!D.empty()) {
            dist_coeffs = cv::Mat::zeros(1, (int)D.size(), CV_64F);
            for (int i = 0; i < (int)D.size(); ++i) dist_coeffs.at<double>(0, i) = D[i];
        }
    } catch (...) { MV_LOG_ERROR() << "Calibration load failed."; }

    std::vector<Metavision::EventSourceId> source_id_events;
    std::vector<Metavision::EventActiveTrack> active_tracks;
    size_t event_counter = 0;
    Metavision::timestamp last_output_ts = 0;

    auto decoder = camera.get_device().get_facility<Metavision::I_EventsStreamDecoder>();
    if (decoder)
        decoder->add_time_callback([&](Metavision::timestamp t) { tracker.notify_elapsed_time(t); });

    camera.cd().add_callback([&](const auto begin, const auto end) {
        source_id_events.clear();
        modulated_light_detector.process_events(begin, end, std::back_inserter(source_id_events));
        active_tracks.clear();
        tracker.process_events(source_id_events.cbegin(), source_id_events.cend(), std::back_inserter(active_tracks));

        event_counter += std::distance(source_id_events.cbegin(), source_id_events.cend());
        Metavision::timestamp current_ts = 0;
        for (const auto &track : active_tracks) current_ts = std::max(current_ts, track.t);
        if (current_ts == 0 && !source_id_events.empty()) current_ts = source_id_events.back().t;

        if (event_counter >= opt_config->delta_n_events) {
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
                if (solvePnP_stable(pts3d, pts2d, camera_matrix, dist_coeffs, rvec, tvec)) {
                    cv::Mat R;
                    cv::Rodrigues(rvec, R);

                    cv::Mat R_wc = R.t();
                    cv::Mat cam_center = -R.t() * tvec;

                    csv_file << current_ts << ","
                             << cam_center.at<double>(0) << ","
                             << cam_center.at<double>(1) << ","
                             << cam_center.at<double>(2) << ",";
                    for (int i = 0; i < 3; ++i)
                        for (int j = 0; j < 3; ++j)
                            csv_file << R_wc.at<double>(i, j)
                                     << ((i == 2 && j == 2) ? "\n" : ",");
                    csv_file.flush();
                }
                last_output_ts = current_ts;
                event_counter = 0;
            } else event_counter = 0;
        }
    });

    camera.start();
    while (camera.is_running()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    camera.stop();

    csv_file.close();
    return 0;
}
