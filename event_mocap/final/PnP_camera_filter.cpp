#include <thread>
#include <chrono>
#include <optional>
#include <fstream>
#include <set>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <cmath>

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

    cv::Mat prev_rvec;   // 직전 프레임의 회전벡터 (camera→world)
    cv::Mat prev_tvec;   // 직전 프레임의 위치 (camera position in world)
    bool has_prev_pose = false;

    // 튀는 해를 제거하기 위한 임계값 (내적)
    const double ROT_DOT_THRESHOLD = 0.5; // 필요시 조절
    // 최근 관측을 얼마(마이크로초 단위)까지 허용할지 (조절 가능)
    const Metavision::timestamp max_age_us = 100000; // 2000 us
    // ID별로 보관할 최대 샘플 수 (메모리용)
    const size_t max_samples_per_id = 10;

    // map: id -> vector of (2D point, timestamp)
    std::unordered_map<std::uint32_t, std::vector<std::pair<cv::Point2f, Metavision::timestamp>>> obs_buffer;

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

    const auto active_marker = detail::load_active_marker(opt_config->am_json_path);

    const auto eb_w = static_cast<std::uint16_t>(camera.geometry().width());
    const auto eb_h = static_cast<std::uint16_t>(camera.geometry().height());
    opt_config->detector_params.width  = eb_w;
    opt_config->detector_params.height = eb_h;

    opt_config->detector_params.tolerance = 0.3f;
    opt_config->detector_params.base_period_us = 400;  

    Metavision::ModulatedLightDetectorAlgorithm modulated_light_detector(
        opt_config->detector_params);

    std::set<std::uint32_t> led_ids;
    for (const auto &kv : active_marker) {
        led_ids.insert(kv.first);
    }

    Metavision::ActiveMarkerTrackerAlgorithm tracker(opt_config->tracker_params, led_ids);

    // calibration 로딩 (원본과 동일)
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

    std::ofstream pose_file("pose_log.csv");
    pose_file << "timestamp,x,y,z,rvec_x,rvec_y,rvec_z\n";

    auto decoder = camera.get_device().get_facility<Metavision::I_EventsStreamDecoder>();
    decoder->add_time_callback([&](Metavision::timestamp t) {
        tracker.notify_elapsed_time(t);
    });

    // Event callback
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
        if (current_ts == 0) {
            // 아무 타임스탬프도 없다면 건너뜀
            return;
        }

        // --- 관측 버퍼 업데이트 ---
        for (const auto &track : active_tracks) {
            auto &vec = obs_buffer[track.id];
            vec.emplace_back(cv::Point2f(track.x, track.y), track.t);
            // 안전을 위해 버퍼 크기 제한
            if (vec.size() > max_samples_per_id) {
                vec.erase(vec.begin(), vec.begin() + (vec.size() - max_samples_per_id));
            }
        }
        // prune 오래된 관측
        for (auto it = obs_buffer.begin(); it != obs_buffer.end();) {
            auto &vec = it->second;
            // 제거 기준: current_ts - t > max_age_us
            vec.erase(std::remove_if(vec.begin(), vec.end(),
                                     [&](const std::pair<cv::Point2f, Metavision::timestamp> &p) {
                                         return (current_ts > p.second) ? (current_ts - p.second > max_age_us) : false;
                                     }),
                      vec.end());
            if (vec.empty()) {
                it = obs_buffer.erase(it);
            } else {
                ++it;
            }
        }

        bool time_ok = (time_threshold > 0 && current_ts - last_output_ts >= time_threshold);
        bool count_ok = (event_threshold > 0 && event_counter >= event_threshold);
        if ((time_threshold == 0 && event_threshold == 0) || time_ok || count_ok) {
            // Build unique 2D-3D correspondences: id당 하나의 2D만 사용 (시간가중 평균)
            std::vector<cv::Point3f> pts3d;
            std::vector<cv::Point2f> pts2d;

            for (const auto &kv : active_marker) {
                const auto id = kv.first;
                auto it = obs_buffer.find(id);
                if (it == obs_buffer.end()) continue;

                const auto &samples = it->second;
                if (samples.empty()) continue;

                // 시간 가중 평균: 최근 관측에 더 가중치
                double sw = 0.0, sx = 0.0, sy = 0.0;
                for (const auto &p : samples) {
                    Metavision::timestamp dt = (current_ts > p.second) ? (current_ts - p.second) : 0;
                    // 가중치 함수: 1 / (1 + dt) -> dt가 작을수록 가중치 큼
                    double w = 1.0 / (1.0 + static_cast<double>(dt)); // dt in us, scales automatically
                    sx += p.first.x * w;
                    sy += p.first.y * w;
                    sw += w;
                }
                if (sw <= 0) continue;
                cv::Point2f avg_pt(static_cast<float>(sx / sw), static_cast<float>(sy / sw));
                pts3d.push_back(kv.second);
                pts2d.push_back(avg_pt);
            }

            if (pts3d.size() >= 4) {
                // solvePnPRansac 로 이상치에 강인하게 계산
                cv::Mat rvec, tvec;
                std::vector<int> inliers;
                bool success = cv::solvePnPRansac(pts3d, pts2d, camera_matrix, dist_coeffs,
                                                  rvec, tvec, false,
                                                  100,       // iterations
                                                  8.0,       // reprojection error threshold (pixels) - 필요시 조정
                                                  0.99,      // confidence
                                                  inliers, cv::SOLVEPNP_ITERATIVE);
                if (!success) {
                    // fallback: 일반 solvePnP (optional)
                    success = cv::solvePnP(pts3d, pts2d, camera_matrix, dist_coeffs, rvec, tvec);
                }

                if (success) {
                    // world→camera R, t 를 camera→world 로 변환
                    cv::Mat R;
                    cv::Rodrigues(rvec, R);   // R = R_cw (world→cam)
                
                    cv::Mat R_wc = R.t();      
                    cv::Mat C_w = -R_wc * tvec;  
                
                    cv::Mat c_rvec;
                    cv::Rodrigues(R_wc, c_rvec);  // camera→world rvec
                
                    // ==========================
                    //       POST FILTER
                    // ==========================
                
                    bool accept = true;
                
                    if (has_prev_pose) {
                        // prev_rvec 과 비교하여 내적 계산
                        cv::Mat v1 = prev_rvec / cv::norm(prev_rvec);
                        cv::Mat v2 = c_rvec    / cv::norm(c_rvec);
                
                        double dot = v1.dot(v2);
                
                        // 튀는 회전이면 제외
                        if (dot < ROT_DOT_THRESHOLD) {
                            accept = false;
                        }
                
                        // 위치도 비정상적으로 튀었는지 추가 체크할 수 있음 (옵션)
                        // double dist = cv::norm(C_w - prev_tvec);
                        // if (dist > MAX_JUMP) accept = false;
                    }
                
                    if (!accept) {
                        // 튄 값이라면 current pose를 건너뛰고 이전 pose를 그대로 출력하거나 skip할 수 있음
                        // 여기서는 "이전 pose 그대로 저장" 옵션
                        cv::Mat c_rvec_use = prev_rvec.clone();
                        cv::Mat C_w_use    = prev_tvec.clone();
                
                        pose_file << current_ts << ","
                                  << C_w_use.at<double>(0) << ","
                                  << C_w_use.at<double>(1) << ","
                                  << C_w_use.at<double>(2) << ","
                                  << c_rvec_use.at<double>(0) << ","
                                  << c_rvec_use.at<double>(1) << ","
                                  << c_rvec_use.at<double>(2) << "\n";
                
                    } else {
                        // 정상값이면 그대로 기록
                        pose_file << current_ts << ","
                                  << C_w.at<double>(0) << ","
                                  << C_w.at<double>(1) << ","
                                  << C_w.at<double>(2) << ","
                                  << c_rvec.at<double>(0) << ","
                                  << c_rvec.at<double>(1) << ","
                                  << c_rvec.at<double>(2) << "\n";
                
                        // prev 갱신
                        prev_rvec = c_rvec.clone();
                        prev_tvec = C_w.clone();
                        has_prev_pose = true;
                    }
                }                
                
                // Reset after solving
                last_output_ts = current_ts;
                event_counter = 0;
            } else {
                // Not enough distinct ids: 이벤트 카운터만 리셋 (원하면 유지도 가능)
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
