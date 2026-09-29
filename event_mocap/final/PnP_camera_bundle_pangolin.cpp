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
#include <cmath>
#include <algorithm>

#include <pangolin/pangolin.h>
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


// ------------------------------
// 글로벌 pose 공유 변수
// ------------------------------
std::mutex pose_mutex;
cv::Mat global_rvec = cv::Mat::zeros(3,1,CV_64F);
cv::Mat global_tvec = cv::Mat::zeros(3,1,CV_64F);
std::atomic<bool> new_pose_available{false};

// ------------------------------
// 원점 축(axis) 표시 함수 추가
// ------------------------------
void drawOriginAxis(float axis_len = 200.0f)
{
    glLineWidth(5.0f);

    glBegin(GL_LINES);

    // X축 (빨강)
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(0,0,0);
    glVertex3f(axis_len,0,0);

    // Y축 (초록)
    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(0,0,0);
    glVertex3f(0,axis_len,0);

    // Z축 (파랑)
    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(0,0,0);
    glVertex3f(0,0,axis_len);

    glEnd();
}

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

// ============================================================================
// Pangolin 시각화 스레드
// ============================================================================
void runPangolinViewer(std::atomic<bool> &run_flag)
{
    pangolin::CreateWindowAndBind("Pose Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.05f, 0.05f, 0.05f, 1.0f);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024,768, 500,500, 512,389, 0.1,5000),
        pangolin::ModelViewLookAt(1000,1000,-2000, 0,0,0, pangolin::AxisZ)
    );

    pangolin::Handler3D handler(s_cam);
    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0,1,0,1,-1024.0f/768.0f)
        .SetHandler(&handler);

    std::vector<cv::Point3d> trajectory;

    while (!pangolin::ShouldQuit() && run_flag.load()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);

        // 원점 Axis (새로 추가)
        drawOriginAxis(80.0f);
        drawGroundGrid(2000.0f, 40);

        // 현재 pose 가져오기
        cv::Mat rvec, tvec;
        {
            std::lock_guard<std::mutex> lock(pose_mutex);
            if (new_pose_available.load()) {
                rvec = global_rvec.clone();
                tvec = global_tvec.clone();
            }
        }

        if (!rvec.empty()) {
            cv::Mat R; cv::Rodrigues(rvec, R);
            cv::Mat cam_center = -R.t() * tvec;

            double x = cam_center.at<double>(0);
            double y = cam_center.at<double>(1);
            double z = cam_center.at<double>(2);
            trajectory.emplace_back(x,y,z);

            // OpenGL pose 행렬 구성
            cv::Mat R_wc = R.t();
            pangolin::OpenGlMatrix T_wc;
            T_wc.SetIdentity();

            for (int i=0;i<3;i++)
                for (int j=0;j<3;j++)
                    T_wc.m[j*4 + i] = R_wc.at<double>(i,j);

            T_wc.m[12] = x;
            T_wc.m[13] = y;
            T_wc.m[14] = z;

            glPushMatrix();
            glMultMatrixd(T_wc.m);
            pangolin::glDrawAxis(100);
            glPopMatrix();
        }

        // 경로 표시
        if (trajectory.size() >= 2) {
            glLineWidth(2.0f);
            glColor3f(0,1,0);
            glBegin(GL_LINE_STRIP);
            for (auto &p : trajectory)
                glVertex3f(p.x,p.y,p.z);
            glEnd();
        }

        pangolin::FinishFrame();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}


int main(int argc, char *argv[]) {
    // --------------------------
    // 1. Config 파싱
    // --------------------------
    auto opt_config = parse_command_line(argc, argv);
    if (!opt_config) return 1;

    // --------------------------
    // 2. 카메라 로드
    // --------------------------
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
            Metavision::FileConfigHints().real_time_playback(opt_config->realtime_playback_speed);
        camera = Metavision::Camera::from_file(opt_config->event_file_path, cam_config);
    }

    // --------------------------
    // 3. Active marker JSON 로드
    // --------------------------
    const auto active_marker = detail::load_active_marker(opt_config->am_json_path);

    // --------------------------
    // 4. Detector/Tracker 설정
    // --------------------------
    const auto eb_w = static_cast<std::uint16_t>(camera.geometry().width());
    const auto eb_h = static_cast<std::uint16_t>(camera.geometry().height());

    opt_config->detector_params.width  = eb_w;
    opt_config->detector_params.height = eb_h;
    opt_config->detector_params.tolerance = 0.3f;
    opt_config->detector_params.base_period_us = 400;

    Metavision::ModulatedLightDetectorAlgorithm modulated_light_detector(opt_config->detector_params);

    std::set<std::uint32_t> led_ids;
    for (const auto &kv : active_marker)
        led_ids.insert(kv.first);

    Metavision::ActiveMarkerTrackerAlgorithm tracker(opt_config->tracker_params, led_ids);

    // --------------------------
    // 5. Calibration 로드
    // --------------------------
    cv::Mat K = cv::Mat::eye(3,3,CV_64F);
    cv::Mat dist_coeffs;

    try {
        namespace pt = boost::property_tree;
        pt::ptree root;
        pt::read_json(opt_config->calib_json_path, root);

        auto intrinsics = root.get_child_optional("intrinsics").value_or(root);
        std::vector<double> KK, DD;

        for (auto &v : intrinsics.get_child("K"))  KK.push_back(v.second.get_value<double>());
        for (auto &v : intrinsics.get_child("D"))  DD.push_back(v.second.get_value<double>());

        if (KK.size() == 9) {
            K.at<double>(0,0) = KK[0];
            K.at<double>(0,2) = KK[2];
            K.at<double>(1,1) = KK[4];
            K.at<double>(1,2) = KK[5];
        }

        if (!DD.empty()) {
            dist_coeffs = cv::Mat::zeros(1, (int)DD.size(), CV_64F);
            for (size_t i = 0; i < DD.size(); i++)
                dist_coeffs.at<double>(0,i) = DD[i];
        }
    } catch (...) {
        MV_LOG_ERROR() << "Calibration load failed!";
    }

    // --------------------------
    // 6. pose estimation timing 설정
    // --------------------------
    std::vector<Metavision::EventSourceId> source_id_events;
    std::vector<Metavision::EventActiveTrack> active_tracks;

    size_t event_counter = 0;
    Metavision::timestamp last_output_ts = 0;

    size_t event_threshold = opt_config->delta_n_events;
    Metavision::timestamp time_threshold = opt_config->delta_ts;

    if (opt_config->estimation_rate_hz > 0) {
        time_threshold = (Metavision::timestamp)(1e6f / opt_config->estimation_rate_hz);
        event_threshold = 0;
    }

    // --------------------------
    // 7. Pangolin 시각화 스레드 시작
    // --------------------------
    std::atomic<bool> run_flag{true};
    std::thread pangolin_thread(runPangolinViewer, std::ref(run_flag));

    // --------------------------
    // 8. Event-stream decoder
    // --------------------------
    auto decoder = camera.get_device().get_facility<Metavision::I_EventsStreamDecoder>();
    if (decoder) {
        decoder->add_time_callback([&](Metavision::timestamp t){
            tracker.notify_elapsed_time(t);
        });
    }

    // --------------------------
    // 9. Event callback (PnP + BA)
    // --------------------------
    camera.cd().add_callback([&](const auto begin, const auto end) {
        source_id_events.clear();
        modulated_light_detector.process_events(begin, end, std::back_inserter(source_id_events));

        active_tracks.clear();
        tracker.process_events(source_id_events.cbegin(), source_id_events.cend(),
                               std::back_inserter(active_tracks));

        event_counter += source_id_events.size();
        Metavision::timestamp current_ts = 0;

        for (const auto &track : active_tracks)
            current_ts = std::max(current_ts, track.t);

        if (current_ts == 0 && !source_id_events.empty())
            current_ts = source_id_events.back().t;

        bool time_ok  = (time_threshold > 0 && current_ts - last_output_ts >= time_threshold);
        bool count_ok = (event_threshold > 0 && event_counter >= event_threshold);

        if (!time_ok && !count_ok) return;

        // ----------------------
        // Collect pts3d/pts2d
        // ----------------------
        std::vector<cv::Point3f> pts3d;
        std::vector<cv::Point2f> pts2d;

        for (const auto &track : active_tracks) {
            auto it = active_marker.find(track.id);
            if (it == active_marker.end()) continue;
            pts3d.push_back(it->second);
            pts2d.emplace_back(track.x, track.y);
        }

        if (pts3d.size() < 4) {
            event_counter = 0;
            return;
        }

        // ----------------------
        // SolvePnPRansac
        // ----------------------
        cv::Mat rvec, tvec;
        std::vector<int> inliers;

        bool success = cv::solvePnPRansac(
            pts3d, pts2d, K, dist_coeffs,
            rvec, tvec, false,
            100, 4.0, 0.99,
            inliers, cv::SOLVEPNP_ITERATIVE
        );

        if (!success) {
            success = cv::solvePnP(pts3d, pts2d, K, dist_coeffs, rvec, tvec);
        }
        if (!success) {
            event_counter = 0;
            return;
        }

        // ----------------------
        // BA refine (카메라 포즈-only)
        // ----------------------
        solvePnP_and_refine_with_BA(pts3d, pts2d, K, dist_coeffs, rvec, tvec);

        // ----------------------
        // rvec/tvec → 월드 좌표계 camera center 변환
        // ----------------------
        cv::Mat R;
        cv::Rodrigues(rvec, R);

        cv::Mat R_wc = R.t();
        cv::Mat cam_center = -R_wc * tvec;

        cv::Mat c_rvec;
        cv::Rodrigues(R_wc, c_rvec);

        // ----------------------
        // Pangolin 표시용 pose 갱신
        // ----------------------
        {
            std::lock_guard<std::mutex> lock(pose_mutex);
            global_rvec = c_rvec.clone();
            global_tvec = cam_center.clone();
            new_pose_available.store(true);
        }

        last_output_ts = current_ts;
        event_counter = 0;
    });

    // --------------------------
    // 10. 카메라 시작
    // --------------------------
    camera.start();

    while (camera.is_running()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    camera.stop();

    // --------------------------
    // 11. 종료 처리
    // --------------------------
    run_flag.store(false);
    if (pangolin_thread.joinable())
        pangolin_thread.join();

    return 0;
}
