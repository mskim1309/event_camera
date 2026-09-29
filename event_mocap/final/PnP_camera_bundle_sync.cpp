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
};

std::optional<Config> parse_command_line(int argc, char *argv[]) {
    namespace po = boost::program_options;
    const std::string program_desc("Active marker 3D tracking with PnP + pose-only BA (bundle)");

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
// solvePnP + pose-only BA (Ceres) refine
// -----------------------
static bool pnp_initialized = false;
static cv::Mat rvec_prev, tvec_prev;

bool solvePnP_and_refine_with_BA(
        const std::vector<cv::Point3f>& pts3d,
        const std::vector<cv::Point2f>& pts2d,
        const cv::Mat& K,
        const cv::Mat& dist,
        cv::Mat& rvec,
        cv::Mat& tvec)
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

    // Prepare parameters for Ceres
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
    options.num_threads = 1;
    options.max_num_iterations = 50;
    options.function_tolerance = 1e-8;
    options.parameter_tolerance = 1e-8;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    // optional: std::cout << summary.BriefReport() << std::endl;

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
    const Metavision::timestamp max_age_us = 50000; // keep recent observations
    const size_t max_samples_per_id = 10;

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
        // --- synchronized sampling + interpolation block ---
        // Hyperparameters (tweak these as needed)
        const Metavision::timestamp sample_interval_us = (opt_config->estimation_rate_hz > 0)
            ? static_cast<Metavision::timestamp>(1000000.0f / opt_config->estimation_rate_hz)
            : 0; // 0 = use current_ts (no enforced fixed interval)

        const Metavision::timestamp max_interp_gap_us = 10000; // max allowed gap for interpolation (10 ms). tune as needed
        const size_t min_ids_required = 4; // minimum number of ids needed to run PnP

        // Helper: interpolate a single id's samples to target_ts
        auto interpolate_for_timestamp = [&](const std::vector<std::pair<cv::Point2f, Metavision::timestamp>>& samples,
                                                Metavision::timestamp target_ts,
                                                cv::Point2f &out_pt) -> bool {
            if (samples.empty()) return false;
            // if target outside range, allow nearest if within max_interp_gap_us (or reject)
            // find index of last sample <= target_ts
            int idx_le = -1;
            for (int i = (int)samples.size()-1; i >= 0; --i) {
                if (samples[i].second <= target_ts) { idx_le = i; break; }
            }
            int idx_ge = -1;
            for (size_t i = 0; i < samples.size(); ++i) {
                if (samples[i].second >= target_ts) { idx_ge = (int)i; break; }
            }
            // exact match
            if (idx_le != -1 && samples[idx_le].second == target_ts) {
                out_pt = samples[idx_le].first;
                return true;
            }
            // if both sides exist -> linear interpolate
            if (idx_le != -1 && idx_ge != -1 && idx_le != idx_ge) {
                auto t0 = samples[idx_le].second;
                auto t1 = samples[idx_ge].second;
                double dt = double(t1 - t0);
                if (dt <= 0.0) return false;
                double alpha = double(target_ts - t0) / dt;
                // check gaps
                if ((target_ts - t0) > max_interp_gap_us || (t1 - target_ts) > max_interp_gap_us) return false;
                const cv::Point2f &p0 = samples[idx_le].first;
                const cv::Point2f &p1 = samples[idx_ge].first;
                out_pt.x = static_cast<float>((1.0 - alpha) * p0.x + alpha * p1.x);
                out_pt.y = static_cast<float>((1.0 - alpha) * p0.y + alpha * p1.y);
                return true;
            }
            // only left exists -> allow nearest if within gap
            if (idx_le != -1 && idx_ge == -1) {
                if (target_ts - samples[idx_le].second <= max_interp_gap_us) {
                    out_pt = samples[idx_le].first;
                    return true;
                }
                return false;
            }
            // only right exists -> allow nearest if within gap
            if (idx_le == -1 && idx_ge != -1) {
                if (samples[idx_ge].second - target_ts <= max_interp_gap_us) {
                    out_pt = samples[idx_ge].first;
                    return true;
                }
                return false;
            }
            return false;
        };

        // decide target timestamp for this PnP run
        Metavision::timestamp target_ts = current_ts;
        if (sample_interval_us > 0) {
            if (last_output_ts == 0) {
                // first output: snap down to nearest sample interval at or before current_ts
                target_ts = current_ts;
            } else {
                Metavision::timestamp candidate = last_output_ts + sample_interval_us;
                if (candidate <= current_ts) target_ts = candidate;
                else {
                    // not yet reached next scheduled sample time -> skip PnP now
                    // we still reset event_counter to avoid bursting
                    // (alternatively you can keep accumulating until candidate <= current_ts)
                    return;
                }
            }
        } else {
            // sample_interval_us == 0 -> use current_ts (as before)
            target_ts = current_ts;
        }

        // Build synchronized pts3d / pts2d by interpolating each id to target_ts
        std::vector<cv::Point3f> pts3d_sync;
        std::vector<cv::Point2f> pts2d_sync;
        pts3d_sync.reserve(active_marker.size());
        pts2d_sync.reserve(active_marker.size());

        for (const auto &kv : active_marker) {
            const auto id = kv.first;
            auto it = obs_buffer.find(id);
            if (it == obs_buffer.end()) continue;
            const auto &samples = it->second;
            cv::Point2f interp_pt;
            bool ok_interp = interpolate_for_timestamp(samples, target_ts, interp_pt);
            if (!ok_interp) continue; // skip this id
            pts3d_sync.push_back(kv.second);
            pts2d_sync.push_back(interp_pt);
        }

        // require minimum number of ids to run PnP
        if (pts3d_sync.size() >= min_ids_required) {
            // run PnP on synchronized observations
            cv::Mat rvec, tvec;
            std::vector<int> inliers;
            bool success = cv::solvePnPRansac(pts3d_sync, pts2d_sync, camera_matrix, dist_coeffs,
                                                rvec, tvec, false,
                                                100, 4.0, 0.99, inliers, cv::SOLVEPNP_ITERATIVE);
            if (!success) {
                success = cv::solvePnP(pts3d_sync, pts2d_sync, camera_matrix, dist_coeffs, rvec, tvec);
            }

            if (success) {
                // you can still call your BA refine here with pts3d_sync/pts2d_sync
                if (!solvePnP_and_refine_with_BA(pts3d_sync, pts2d_sync, camera_matrix, dist_coeffs, rvec, tvec)) {
                    MV_LOG_WARNING() << "Pose refinement failed; using solvePnP result.";
                }

                // produce and log pose (same as before)
                cv::Mat R; cv::Rodrigues(rvec, R);
                cv::Mat R_wc = R.t();
                cv::Mat C_w = -R_wc * tvec;
                cv::Mat c_rvec; cv::Rodrigues(R_wc, c_rvec);

                double cx = C_w.at<double>(0);
                double cy = C_w.at<double>(1);
                double cz = C_w.at<double>(2);
                double rvec_x = c_rvec.at<double>(0);
                double rvec_y = c_rvec.at<double>(1);
                double rvec_z = c_rvec.at<double>(2);

                pose_file << target_ts << "," << cx << "," << cy << "," << cz << "," << rvec_x << "," << rvec_y << "," << rvec_z << "\n";

                // update pacing variables
                last_output_ts = target_ts;
                event_counter = 0;
            } else {
                // PnP failed despite synchronized samples
                event_counter = 0;
            }
        } else {
            // not enough synchronized ids -> do nothing (keep accumulating)
            // optionally you can force PnP if pts3d_sync.size() >= some lower bound
            event_counter = 0;
        }

    });

    camera.start();
    while (camera.is_running()) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    camera.stop();

    return 0;
}
