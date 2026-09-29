#include <Eigen/LU>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/program_options.hpp>

#include <metavision/hal/facilities/i_events_stream_decoder.h>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/cv/events/event_source_id.h>
#include <metavision/sdk/cv/utils/camera_geometry_factory.h>
#include <metavision/sdk/cv/algorithms/modulated_light_detector_algorithm.h>
#include <metavision/sdk/cv3d/algorithms/active_marker_pose_estimator_algorithm.h>
#include <metavision/sdk/driver/camera.h>

#include <thread>
#include <optional>

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
    Metavision::ActiveMarkerPoseEstimatorAlgorithm::Params pose_estimator_params;
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
        ("tracker-update-radius", po::value<bool>(&config.pose_estimator_params.tracker_params.update_radius)->default_value(true))
        ("tracker-inactivity-period-us", po::value<Metavision::timestamp>(&config.pose_estimator_params.tracker_params.inactivity_period_us)->default_value(1000))
        ("tracker-monitoring-frequency", po::value<float>(&config.pose_estimator_params.tracker_params.monitoring_frequency_hz)->default_value(30.f))
        ("tracker-radius2", po::value<float>(&config.pose_estimator_params.tracker_params.radius)->default_value(30.f))
        ("tracker-distance-percentage", po::value<float>(&config.pose_estimator_params.tracker_params.distance_pct)->default_value(0.3f))
        ("tracker-alpha-pos", po::value<float>(&config.pose_estimator_params.tracker_params.alpha_pos)->default_value(0.05f));

    int pose_estimation_type_int = 0;
    po::options_description pose_estimation_options("Pose estimation options");
    pose_estimation_options.add_options()
        ("pose-estimation-type", po::value<int>(&pose_estimation_type_int)->default_value(0), "Estimator type (0: CameraClock, 1: Other)")
        ("pose-estimation-n-events", po::value<size_t>(&config.pose_estimator_params.delta_n_events)->default_value(5000))
        ("pose-estimation-n-us", po::value<Metavision::timestamp>(&config.pose_estimator_params.delta_ts)->default_value(0))
        ("pose-estimation-frequency-hz", po::value<float>(&config.pose_estimator_params.estimation_rate_hz)->default_value(0.f));

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
    config.pose_estimator_params.type = static_cast<Metavision::ActiveMarkerPoseEstimatorAlgorithm::EstimatorType>(pose_estimation_type_int);

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
    if (!opt_config)
        return 1;

    Metavision::Camera camera;
    if (opt_config->event_file_path.empty()) {
        Metavision::DeviceConfig device_config;
        device_config.enable_biases_range_check_bypass(true);
        camera = Metavision::Camera::from_first_available(device_config);
        camera.load(opt_config->cam_config_path);
    } else {
        const auto cam_config = Metavision::FileConfigHints().real_time_playback(opt_config->realtime_playback_speed);
        camera = Metavision::Camera::from_file(opt_config->event_file_path, cam_config);
    }

    const auto active_marker = detail::load_active_marker(opt_config->am_json_path);
    const auto camera_geometry = Metavision::load_camera_geometry<float>(opt_config->calib_json_path);

    const auto eb_w = static_cast<std::uint16_t>(camera.geometry().width());
    const auto eb_h = static_cast<std::uint16_t>(camera.geometry().height());
    opt_config->detector_params.width  = eb_w;
    opt_config->detector_params.height = eb_h;

    Metavision::ModulatedLightDetectorAlgorithm modulated_light_detector(opt_config->detector_params);
    Metavision::ActiveMarkerPoseEstimatorAlgorithm pose_estimator(opt_config->pose_estimator_params, *camera_geometry, active_marker);

    auto decoder = camera.get_device().get_facility<Metavision::I_EventsStreamDecoder>();
    decoder->add_time_callback([&](Metavision::timestamp t) { pose_estimator.notify_elapsed_time(t); });

    std::vector<Metavision::EventSourceId> source_id_events;
    camera.cd().add_callback([&](const auto begin, const auto end) {
        source_id_events.clear();
        modulated_light_detector.process_events(begin, end, std::back_inserter(source_id_events));
        pose_estimator.process_events(source_id_events.cbegin(), source_id_events.cend());
    });

    pose_estimator.set_pose_update_callback(
        [&](Metavision::timestamp t, const std::optional<Eigen::Matrix4f> &T_w_m_opt) {
            if (T_w_m_opt) {
                std::cout << "Time: " << t << " Pose:\n" << *T_w_m_opt << std::endl;
            } else {
                std::cout << "Time: " << t << " Pose not available." << std::endl;
            }
        });

    camera.start();
    while (camera.is_running()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    camera.stop();

    return 0;
}
