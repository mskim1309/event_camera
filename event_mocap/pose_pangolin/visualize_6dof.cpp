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

#include <pangolin/pangolin.h>
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
    const std::string program_desc("Active marker 3D tracking with Pangolin visualization");

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
    if (config.event_file_path.empty() && config.cam_config_path.empty()) {
        MV_LOG_ERROR() << "Camera config file is required for live execution";
        return std::nullopt;
    }

    return config;
}

// ======================
// Ground Grid Renderer
// ======================
void drawGroundGrid(float grid_size = 1000.0f, int num_divisions = 20) {
    glLineWidth(1.0f);
    glColor3f(0.85f, 0.85f, 0.85f); // 옅은 회색

    float half_size = grid_size / 2.0f;
    float step = grid_size / num_divisions;

    glBegin(GL_LINES);
    for (int i = 0; i <= num_divisions; ++i) {
        float pos = -half_size + i * step;
        // X 방향 선
        glVertex3f(-half_size, pos, 0);
        glVertex3f(half_size, pos, 0);
        // Y 방향 선
        glVertex3f(pos, -half_size, 0);
        glVertex3f(pos, half_size, 0);
    }
    glEnd();
}

int main(int argc, char *argv[]) {
    auto opt_config = parse_command_line(argc, argv);
    if (!opt_config) return 1;

    // Camera setup
    Metavision::Camera camera;
    if (opt_config->event_file_path.empty()) {
        Metavision::DeviceConfig device_config;
        device_config.enable_biases_range_check_bypass(true);
        camera = Metavision::Camera::from_first_available(device_config);
        if (!opt_config->cam_config_path.empty()) camera.load(opt_config->cam_config_path);
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

    // Camera intrinsics load
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

    // Event loop setup
    std::vector<Metavision::EventSourceId> source_id_events;
    std::vector<Metavision::EventActiveTrack> active_tracks;
    size_t event_counter = 0;
    Metavision::timestamp last_output_ts = 0;
    size_t event_threshold = opt_config->delta_n_events;
    Metavision::timestamp time_threshold = opt_config->delta_ts;
    if (opt_config->estimation_rate_hz > 0) {
        time_threshold = static_cast<Metavision::timestamp>(1e6f / opt_config->estimation_rate_hz);
        event_threshold = 0;
    }

    auto decoder = camera.get_device().get_facility<Metavision::I_EventsStreamDecoder>();
    if (decoder)
        decoder->add_time_callback([&](Metavision::timestamp t) {
            tracker.notify_elapsed_time(t);
        });

    camera.cd().add_callback([&](const auto begin, const auto end) {
        source_id_events.clear();
        modulated_light_detector.process_events(begin, end, std::back_inserter(source_id_events));
        active_tracks.clear();
        tracker.process_events(source_id_events.cbegin(), source_id_events.cend(), std::back_inserter(active_tracks));

        event_counter += std::distance(source_id_events.cbegin(), source_id_events.cend());
        Metavision::timestamp current_ts = 0;
        for (const auto &track : active_tracks) current_ts = std::max(current_ts, track.t);
        if (current_ts == 0 && !source_id_events.empty()) current_ts = source_id_events.back().t;

        bool time_ok = (time_threshold > 0 && current_ts - last_output_ts >= time_threshold);
        bool count_ok = (event_threshold > 0 && event_counter >= event_threshold);

        if ((time_threshold == 0 && event_threshold == 0) || time_ok || count_ok) {
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
                if (cv::solvePnP(pts3d, pts2d, camera_matrix, dist_coeffs, rvec, tvec)) {
                    std::lock_guard<std::mutex> lock(pose_mutex);
                    rvec.copyTo(global_rvec);
                    tvec.copyTo(global_tvec);
                    new_pose_available.store(true);
                }
                last_output_ts = current_ts;
                event_counter = 0;
            } else event_counter = 0;
        }
    });

    camera.start();

    // ======================
    // PANGOLIN Visualization
    // ======================
    pangolin::CreateWindowAndBind("Pose Viewer", 1024, 768);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) std::cerr << "GLEW init error\n";
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.05f, 0.05f, 0.05f, 1.0f);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 420, 420, 512, 389, 0.1, 5000),
        pangolin::ModelViewLookAt(2000, 2000, 3000, 0, 0, 0, pangolin::AxisZ)
    );
    pangolin::Handler3D handler(s_cam);
    pangolin::View &d_cam = pangolin::CreateDisplay().SetBounds(0, 1, 0, 1, -1024.0f/768.0f).SetHandler(&handler);

    std::vector<cv::Point3d> trajectory;

    // Main Loop
    while (!pangolin::ShouldQuit() && camera.is_running()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);

        // Draw ground grid
        drawGroundGrid(2000.0f, 40);

        // Draw pose & trajectory
        cv::Mat rvec, tvec;
        {
            std::lock_guard<std::mutex> lock(pose_mutex);
            if (new_pose_available.load() && !global_rvec.empty()) {
                rvec = global_rvec.clone();
                tvec = global_tvec.clone();
            }
        }

        if (!rvec.empty()) {
            cv::Mat R; cv::Rodrigues(rvec, R);                // R: world->camera
            cv::Mat cam_center = -R.t() * tvec;               // camera center in world coords
            double x = cam_center.at<double>(0), y = cam_center.at<double>(1), z = cam_center.at<double>(2);
            if (std::abs(x) < 5000 && std::abs(y) < 5000 && std::abs(z) < 5000)
                trajectory.emplace_back(x, y, z);

            // Build OpenGL matrix for camera pose in world coordinates (camera -> world)
            cv::Mat R_wc = R.t(); // R_wc: camera->world rotation

            pangolin::OpenGlMatrix T_wc;
            T_wc.SetIdentity();

            // OpenGlMatrix expects column-major order: m[col*4 + row]
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    T_wc.m[j*4 + i] = R_wc.at<double>(i, j); // fill 3x3 rotation
                }
            }
            // set translation (camera center in world)
            for (int i = 0; i < 3; ++i) {
                T_wc.m[12 + i] = cam_center.at<double>(i);
            }

            // Draw the camera frame (axis + box) using T_wc
            glPushMatrix();
            glMultMatrixd(T_wc.m);
            pangolin::glDrawAxis(100.0f);

            // draw a small square (camera body) in the camera frame
            glLineWidth(2.0f);
            glColor3f(1.0f, 0.5f, 0.0f); // orange
            glBegin(GL_LINES);
            glVertex3f(-10, -10, 0); glVertex3f(10, -10, 0);
            glVertex3f(10, -10, 0);  glVertex3f(10, 10, 0);
            glVertex3f(10, 10, 0);   glVertex3f(-10, 10, 0);
            glVertex3f(-10, 10, 0);  glVertex3f(-10, -10, 0);
            glEnd();
            glPopMatrix();
        }

        if (trajectory.size() >= 2) {
            glLineWidth(1.5f);
            glColor3f(0.0f, 1.0f, 0.0f);
            glBegin(GL_LINE_STRIP);
            for (const auto &p : trajectory) glVertex3f(p.x, p.y, p.z);
            glEnd();
        }

        pangolin::FinishFrame();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    if (camera.is_running()) camera.stop();
    return 0;
}
