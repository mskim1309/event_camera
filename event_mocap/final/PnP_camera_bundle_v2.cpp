#include <thread>
#include <chrono>
#include <optional>
#include <fstream>
#include <set>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <deque>

#include <opencv2/opencv.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/program_options.hpp>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

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
    bool realtime_playback_speed = false;
    Metavision::ModulatedLightDetectorAlgorithm::Params detector_params;
    Metavision::ActiveMarkerTrackerAlgorithm::Params tracker_params;
    size_t delta_n_events = 5000;
    Metavision::timestamp delta_ts = 0;
    float estimation_rate_hz = 0.f;
    int ba_window_size = 5; // sliding window size (frames)
};

std::optional<Config> parse_command_line(int argc, char *argv[]) {
    namespace po = boost::program_options;
    const std::string program_desc("Active marker 3D tracking with PnP + sliding-window pose-only BA (Ceres)");

    Config config;
    po::options_description options_desc;
    po::options_description base_options("Base options");
    base_options.add_options()
        ("help,h", "Produce help message.")
        ("input-event-file,i", po::value<std::string>(&config.event_file_path), "Path to input event file (RAW or HDF5). If not specified, camera live stream is used.")
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
        ("pose-estimation-frequency-hz", po::value<float>(&config.estimation_rate_hz)->default_value(0.f))
        ("ba-window-size", po::value<int>(&config.ba_window_size)->default_value(5), "Sliding window size for pose-only BA (frames). Increase for accuracy (costly).");

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
    if (config.ba_window_size < 1) config.ba_window_size = 1;
    return config;
}

// -----------------------
// Ceres reprojection error (supports radial & tangential like original)
// -----------------------
struct ReprojectionError {
    ReprojectionError(const cv::Point3f& point3d,
                      const cv::Point2f& observation,
                      double fx, double fy, double cx, double cy,
                      const std::vector<double>& dist_coeffs)
        : X_(point3d.x), Y_(point3d.y), Z_(point3d.z),
          u_(observation.x), v_(observation.y),
          fx_(fx), fy_(fy), cx_(cx), cy_(cy),
          dist_(dist_coeffs) {}

    template <typename T>
    bool operator()(const T* const angle_axis, const T* const t, T* residuals) const {
        T point[3];
        point[0] = T(X_);
        point[1] = T(Y_);
        point[2] = T(Z_);

        T p[3];
        ceres::AngleAxisRotatePoint(angle_axis, point, p);

        p[0] += t[0];
        p[1] += t[1];
        p[2] += t[2];

        T xp = p[0] / p[2];
        T yp = p[1] / p[2];

        T r2 = xp*xp + yp*yp;
        T radial = T(1.0);

        int n = static_cast<int>(dist_.size());
        if (n >= 1) radial += T(dist_[0]) * r2;
        if (n >= 2) radial += T(dist_[1]) * r2 * r2;
        if (n >= 5) radial += T(dist_[4]) * r2 * r2 * r2;

        T x_dist = xp * radial;
        T y_dist = yp * radial;

        if (n >= 3) {
            T p1 = T(dist_[2]);
            T p2 = (n >= 4) ? T(dist_[3]) : T(0.0);
            x_dist += T(2.0) * p1 * xp * yp + p2 * (r2 + T(2.0) * xp * xp);
            y_dist += p1 * (r2 + T(2.0) * yp * yp) + T(2.0) * p2 * xp * yp;
        }

        T u_proj = T(fx_) * x_dist + T(cx_);
        T v_proj = T(fy_) * y_dist + T(cy_);

        residuals[0] = u_proj - T(u_);
        residuals[1] = v_proj - T(v_);
        return true;
    }

    static ceres::CostFunction* Create(const cv::Point3f& p3d,
                                       const cv::Point2f& obs,
                                       double fx, double fy, double cx, double cy,
                                       const std::vector<double>& dist) {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3, 3>(
            new ReprojectionError(p3d, obs, fx, fy, cx, cy, dist)));
    }

    double X_, Y_, Z_;
    double u_, v_;
    double fx_, fy_, cx_, cy_;
    std::vector<double> dist_;
};

// -----------------------
// solvePnP + sliding-window pose-only BA (Ceres) refine
// -----------------------
static bool pnp_initialized = false;
static cv::Mat rvec_prev, tvec_prev;

struct FrameObservation {
    Metavision::timestamp ts;
    std::vector<std::uint32_t> ids;
    std::vector<cv::Point2f> pts2d; // correspond to ids
};

bool run_sliding_window_ba(
        const std::deque<FrameObservation>& window,
        const detail::ActiveMarker& active_marker,
        const cv::Mat& K,
        const cv::Mat& dist,
        cv::Mat& out_rvec_refined,
        cv::Mat& out_tvec_refined,
        int verbose = 0)
{
    if (window.empty()) return false;

    const int W = static_cast<int>(window.size());
    // parameter storage: angle_axis and trans per frame
    std::vector<double> angle_axis_vec(W * 3);
    std::vector<double> trans_vec(W * 3);

    // initialize parameter blocks: use previous poses if available, else identity/zeros
    // We'll initialize each pose by running solvePnP on that frame's observations if possible
    double fx = K.at<double>(0,0);
    double fy = K.at<double>(1,1);
    double cx = K.at<double>(0,2);
    double cy = K.at<double>(1,2);

    std::vector<double> dist_coeffs;
    if (!dist.empty()) {
        for (int i = 0; i < dist.cols; ++i) dist_coeffs.push_back(dist.at<double>(0, i));
    }

    // For each frame in window, compute an initial pose via solvePnP if enough points, otherwise copy last known
    cv::Mat rvec_init, tvec_init;
    for (int i = 0; i < W; ++i) {
        const auto &fr = window[i];
        std::vector<cv::Point3f> pts3d_f;
        std::vector<cv::Point2f> pts2d_f;
        for (size_t j = 0; j < fr.ids.size(); ++j) {
            auto it = active_marker.find(fr.ids[j]);
            if (it == active_marker.end()) continue;
            pts3d_f.push_back(it->second);
            pts2d_f.push_back(fr.pts2d[j]);
        }
        bool solved = false;
        if (pts3d_f.size() >= 4) {
            rvec_init = cv::Mat::zeros(3,1,CV_64F);
            tvec_init = cv::Mat::zeros(3,1,CV_64F);
            solved = cv::solvePnP(pts3d_f, pts2d_f, K, dist, rvec_init, tvec_init, false, cv::SOLVEPNP_ITERATIVE);
        }
        if (!solved) {
            // fallback: use previous pose if available, else identity
            if (pnp_initialized) {
                rvec_init = rvec_prev.clone();
                tvec_init = tvec_prev.clone();
            } else {
                rvec_init = cv::Mat::zeros(3,1,CV_64F);
                tvec_init = cv::Mat::zeros(3,1,CV_64F);
            }
        }
        angle_axis_vec[3*i + 0] = rvec_init.at<double>(0,0);
        angle_axis_vec[3*i + 1] = rvec_init.at<double>(1,0);
        angle_axis_vec[3*i + 2] = rvec_init.at<double>(2,0);
        trans_vec[3*i + 0] = tvec_init.at<double>(0,0);
        trans_vec[3*i + 1] = tvec_init.at<double>(1,0);
        trans_vec[3*i + 2] = tvec_init.at<double>(2,0);
    }

    // Build Ceres problem: parameter blocks per pose and residuals for all observations in window
    ceres::Problem problem;

    for (int i = 0; i < W; ++i) {
        problem.AddParameterBlock(&angle_axis_vec[3*i], 3);
        problem.AddParameterBlock(&trans_vec[3*i], 3);
    }

    // Add residuals
    for (int i = 0; i < W; ++i) {
        const auto &fr = window[i];
        for (size_t j = 0; j < fr.ids.size(); ++j) {
            auto it = active_marker.find(fr.ids[j]);
            if (it == active_marker.end()) continue; // shouldn't happen
            ceres::CostFunction* cf = ReprojectionError::Create(it->second, fr.pts2d[j], fx, fy, cx, cy, dist_coeffs);
            problem.AddResidualBlock(cf, new ceres::HuberLoss(1.0), &angle_axis_vec[3*i], &trans_vec[3*i]);
        }
    }

    // Solver options tuned for accuracy
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.num_threads = 4;
    options.max_num_iterations = 200; // increase for accuracy
    options.function_tolerance = 1e-12;
    options.parameter_tolerance = 1e-12;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (verbose) std::cout << summary.BriefReport() << std::endl;

    // write out refined last pose (most recent frame)
    int idx_last = W - 1;
    out_rvec_refined = cv::Mat::zeros(3,1,CV_64F);
    out_tvec_refined = cv::Mat::zeros(3,1,CV_64F);
    out_rvec_refined.at<double>(0,0) = angle_axis_vec[3*idx_last + 0];
    out_rvec_refined.at<double>(1,0) = angle_axis_vec[3*idx_last + 1];
    out_rvec_refined.at<double>(2,0) = angle_axis_vec[3*idx_last + 2];
    out_tvec_refined.at<double>(0,0) = trans_vec[3*idx_last + 0];
    out_tvec_refined.at<double>(1,0) = trans_vec[3*idx_last + 1];
    out_tvec_refined.at<double>(2,0) = trans_vec[3*idx_last + 2];

    return true;
}

bool solvePnP_and_refine_with_BA(
        const std::vector<cv::Point3f>& pts3d,
        const std::vector<cv::Point2f>& pts2d,
        const cv::Mat& K,
        const cv::Mat& dist,
        cv::Mat& rvec,
        cv::Mat& tvec,
        const std::deque<FrameObservation>& window_for_ba,
        const detail::ActiveMarker& active_marker)
{
    bool ok = false;
    // Use previous estimate when available (improves stability)
    if (!pnp_initialized) {
        ok = cv::solvePnP(pts3d, pts2d, K, dist, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
        if (!ok) return false;
        rvec_prev = rvec.clone();
        tvec_prev = tvec.clone();
        pnp_initialized = true;
    } else {
        // initialize from previous and allow iterative refine
        rvec = rvec_prev.clone();
        tvec = tvec_prev.clone();
        ok = cv::solvePnP(pts3d, pts2d, K, dist, rvec, tvec, true, cv::SOLVEPNP_ITERATIVE);
        if (!ok) return false;
        rvec_prev = rvec.clone();
        tvec_prev = tvec.clone();
    }

    // If window size > 1, run sliding-window pose-only BA
    cv::Mat rvec_refined, tvec_refined;
    if (!window_for_ba.empty() && window_for_ba.size() > 1) {
        if (!run_sliding_window_ba(window_for_ba, active_marker, K, dist, rvec_refined, tvec_refined)) {
            // fallback to single-pose BA (existing behavior)
            // proceed to run single-pose Ceres BA as before
        } else {
            // use refined results for last pose
            rvec = rvec_refined.clone();
            tvec = tvec_refined.clone();
            rvec_prev = rvec.clone();
            tvec_prev = tvec.clone();
            return true;
        }
    }

    // If we reach here, either window size == 1 or sliding BA failed -> fallback single-pose BA (unchanged)
    double fx = K.at<double>(0,0);
    double fy = K.at<double>(1,1);
    double cx = K.at<double>(0,2);
    double cy = K.at<double>(1,2);

    std::vector<double> dist_coeffs;
    if (!dist.empty()) {
        for (int i = 0; i < dist.cols; ++i) dist_coeffs.push_back(dist.at<double>(0, i));
    }

    double angle_axis[3] = { rvec.at<double>(0,0), rvec.at<double>(1,0), rvec.at<double>(2,0) };
    double trans[3] = { tvec.at<double>(0,0), tvec.at<double>(1,0), tvec.at<double>(2,0) };

    ceres::Problem problem;
    problem.AddParameterBlock(angle_axis, 3);
    problem.AddParameterBlock(trans, 3);

    for (size_t i = 0; i < pts3d.size(); ++i) {
        ceres::CostFunction* cf = ReprojectionError::Create(pts3d[i], pts2d[i], fx, fy, cx, cy, dist_coeffs);
        problem.AddResidualBlock(cf, new ceres::HuberLoss(1.0), angle_axis, trans);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.num_threads = 4;
    options.max_num_iterations = 50;
    options.function_tolerance = 1e-8;
    options.parameter_tolerance = 1e-8;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    rvec.at<double>(0,0) = angle_axis[0];
    rvec.at<double>(1,0) = angle_axis[1];
    rvec.at<double>(2,0) = angle_axis[2];
    tvec.at<double>(0,0) = trans[0];
    tvec.at<double>(1,0) = trans[1];
    tvec.at<double>(2,0) = trans[2];

    rvec_prev = rvec.clone();
    tvec_prev = tvec.clone();

    return true;
}

int main(int argc, char *argv[]) {
    auto opt_config = parse_command_line(argc, argv);
    if (!opt_config) return 1;

    // Parameters
    const Metavision::timestamp max_age_us = 2000; // keep recent observations (us)
    const size_t max_samples_per_id = 50;

    std::unordered_map<std::uint32_t, std::vector<std::pair<cv::Point2f, Metavision::timestamp>>> obs_buffer;

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

    // Load calibration
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
            for (auto &v : intrinsics.get_child("K")) K.push_back(v.second.get_value<double>());
        }
        if (auto dnode = intrinsics.get_child_optional("D")) {
            for (auto &v : intrinsics.get_child("D")) D.push_back(v.second.get_value<double>());
        }
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
    } catch (const std::exception &e) {
        MV_LOG_ERROR() << "Failed to load calibration: " << e.what();
    }

    std::vector<Metavision::EventSourceId> source_id_events;
    std::vector<Metavision::EventActiveTrack> active_tracks;
    size_t event_counter = 0;
    Metavision::timestamp last_output_ts = 0;
    size_t event_threshold = opt_config->delta_n_events;
    Metavision::timestamp time_threshold = opt_config->delta_ts;
    if (opt_config->estimation_rate_hz > 0) {
        time_threshold = static_cast<Metavision::timestamp>(1000000.0f / opt_config->estimation_rate_hz);
        event_threshold = 0;
    }

    std::ofstream pose_file("pose_log.csv");
    pose_file << "timestamp,x,y,z,rvec_x,rvec_y,rvec_z\n";

    auto decoder = camera.get_device().get_facility<Metavision::I_EventsStreamDecoder>();
    if (decoder) decoder->add_time_callback([&](Metavision::timestamp t) { tracker.notify_elapsed_time(t); });

    // sliding window container
    std::deque<FrameObservation> ba_window;
    const int BA_WINDOW_MAX = opt_config->ba_window_size;

    camera.cd().add_callback([&](const auto begin, const auto end) {
        source_id_events.clear();
        modulated_light_detector.process_events(begin, end, std::back_inserter(source_id_events));

        active_tracks.clear();
        tracker.process_events(source_id_events.cbegin(), source_id_events.cend(), std::back_inserter(active_tracks));

        size_t new_events = std::distance(source_id_events.cbegin(), source_id_events.cend());
        event_counter += new_events;

        Metavision::timestamp current_ts = 0;
        for (const auto &track : active_tracks) current_ts = std::max(current_ts, track.t);
        if (current_ts == 0 && !source_id_events.empty()) current_ts = source_id_events.back().t;
        if (current_ts == 0) return;

        // update obs buffer
        for (const auto &track : active_tracks) {
            auto &vec = obs_buffer[track.id];
            vec.emplace_back(cv::Point2f(track.x, track.y), track.t);
            if (vec.size() > max_samples_per_id) vec.erase(vec.begin(), vec.begin() + (vec.size() - max_samples_per_id));
        }
        // prune old
        for (auto it = obs_buffer.begin(); it != obs_buffer.end();) {
            auto &vec = it->second;
            vec.erase(std::remove_if(vec.begin(), vec.end(),
                                     [&](const std::pair<cv::Point2f, Metavision::timestamp> &p) {
                                         return (current_ts > p.second) ? (current_ts - p.second > max_age_us) : false;
                                     }), vec.end());
            if (vec.empty()) it = obs_buffer.erase(it); else ++it;
        }

        bool time_ok = (time_threshold > 0 && current_ts - last_output_ts >= time_threshold);
        bool count_ok = (event_threshold > 0 && event_counter >= event_threshold);
        if ((time_threshold == 0 && event_threshold == 0) || time_ok || count_ok) {
            std::vector<cv::Point3f> pts3d;
            std::vector<cv::Point2f> pts2d;

            // prepare a single-frame observation (time-weighted average) for immediate PnP
            for (const auto &kv : active_marker) {
                const auto id = kv.first;
                auto it = obs_buffer.find(id);
                if (it == obs_buffer.end()) continue;
                const auto &samples = it->second;
                if (samples.empty()) continue;

                // time-weighted average
                double sw=0, sx=0, sy=0;
                for (const auto &p : samples) {
                    Metavision::timestamp dt = (current_ts > p.second) ? (current_ts - p.second) : 0;
                    double w = 1.0 / (1.0 + static_cast<double>(dt));
                    sx += p.first.x * w;
                    sy += p.first.y * w;
                    sw += w;
                }
                if (sw <= 0) continue;
                cv::Point2f avg_pt(static_cast<float>(sx/sw), static_cast<float>(sy/sw));
                pts3d.push_back(kv.second);
                pts2d.push_back(avg_pt);
            }

            if (pts3d.size() >= 4) {
                cv::Mat rvec, tvec;
                std::vector<int> inliers;
                bool success = cv::solvePnPRansac(pts3d, pts2d, camera_matrix, dist_coeffs,
                                                  rvec, tvec, false,
                                                  100, 4.0, 0.99, inliers, cv::SOLVEPNP_ITERATIVE);
                if (!success) {
                    success = cv::solvePnP(pts3d, pts2d, camera_matrix, dist_coeffs, rvec, tvec);
                }

                if (success) {
                    // push current frame to BA window (store ids & averaged pts)
                    FrameObservation fr;
                    fr.ts = current_ts;
                    for (size_t k = 0; k < pts3d.size(); ++k) {
                        // find id by matching 3D position in active_marker (fast enough since active_marker small)
                        // better would be to track id->3D when building pts3d/pts2d but keep simple here
                        for (const auto &kv : active_marker) {
                            if (kv.second == pts3d[k]) { fr.ids.push_back(kv.first); break; }
                        }
                        fr.pts2d.push_back(pts2d[k]);
                    }

                    // keep window length
                    ba_window.push_back(fr);
                    while ((int)ba_window.size() > BA_WINDOW_MAX) ba_window.pop_front();

                    // Run pose estimation + sliding-window BA using the frames in ba_window
                    cv::Mat rvec_refined = rvec.clone();
                    cv::Mat tvec_refined = tvec.clone();
                    if (!solvePnP_and_refine_with_BA(pts3d, pts2d, camera_matrix, dist_coeffs, rvec_refined, tvec_refined, ba_window, active_marker)) {
                        MV_LOG_WARNING() << "Pose refinement failed; using solvePnP result.";
                    }

                    // compute camera center in world frame
                    cv::Mat R;
                    cv::Rodrigues(rvec_refined, R); // R = R_cw (world->cam)
                    cv::Mat R_wc = R.t();
                    cv::Mat C_w = -R_wc * tvec_refined;

                    cv::Mat c_rvec;
                    cv::Rodrigues(R_wc, c_rvec);

                    double cx = C_w.at<double>(0);
                    double cy = C_w.at<double>(1);
                    double cz = C_w.at<double>(2);
                    double rvec_x = c_rvec.at<double>(0);
                    double rvec_y = c_rvec.at<double>(1);
                    double rvec_z = c_rvec.at<double>(2);

                    pose_file << current_ts << "," << cx << "," << cy << "," << cz << "," << rvec_x << "," << rvec_y << "," << rvec_z << "\n";
                }

                last_output_ts = current_ts;
                event_counter = 0;
            } else {
                event_counter = 0;
            }
        }
    });

    camera.start();
    while (camera.is_running()) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    camera.stop();

    return 0;
}
