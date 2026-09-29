project(3d_pose)

cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)

find_package(MetavisionSDK COMPONENTS base cv cv3d driver REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(Pangolin REQUIRED)


set(sample PnP)
add_executable(${sample} ${sample}.cpp)
target_link_libraries(${sample}
    MetavisionSDK::base
    MetavisionSDK::cv
    MetavisionSDK::cv3d
    MetavisionSDK::driver
    Boost::program_options
)


set(sample2 PnP_camera_pose)
add_executable(${sample2} ${sample2}.cpp)
target_link_libraries(${sample2}
    MetavisionSDK::base
    MetavisionSDK::cv
    MetavisionSDK::cv3d
    MetavisionSDK::driver
    Boost::program_options
)


set(sample3 PnP_IPPE_camera)
add_executable(${sample3} ${sample3}.cpp)
target_link_libraries(${sample3}
    MetavisionSDK::base
    MetavisionSDK::cv
    MetavisionSDK::cv3d
    MetavisionSDK::driver
    Boost::program_options
)


set(sample4 PnP_camera_filter)
add_executable(${sample4} ${sample4}.cpp)
target_link_libraries(${sample4}
    MetavisionSDK::base
    MetavisionSDK::cv
    MetavisionSDK::cv3d
    MetavisionSDK::driver
    Boost::program_options
)


set(sample5 PnP_camera_bundle)
add_executable(${sample5} ${sample5}.cpp)
target_link_libraries(${sample5}
    MetavisionSDK::base
    MetavisionSDK::cv
    MetavisionSDK::cv3d
    MetavisionSDK::driver
    Boost::program_options
    ${CERES_LIBRARIES}
    Eigen3::Eigen
)


set(sample6 PnP_camera_bundle_sync)
add_executable(${sample6} ${sample6}.cpp)
target_link_libraries(${sample6}
    MetavisionSDK::base
    MetavisionSDK::cv
    MetavisionSDK::cv3d
    MetavisionSDK::driver
    Boost::program_options
    ${CERES_LIBRARIES}
    Eigen3::Eigen
)


set(sample7 PnP_camera_bundle_pangolin)
add_executable(${sample7} ${sample7}.cpp)
target_link_libraries(${sample7}
    MetavisionSDK::base
    MetavisionSDK::cv
    MetavisionSDK::cv3d
    MetavisionSDK::driver
    Boost::program_options
    ${CERES_LIBRARIES}
    Eigen3::Eigen
    
    ${OpenCV_LIBS}
    ${OPENGL_LIBRARIES}
    GLEW::GLEW
    pango_core
    pango_opengl
    pango_windowing
    pango_scene
    pango_plot
)


set(sample8 PnP_camera_bundle_v2)
add_executable(${sample8} ${sample8}.cpp)
target_link_libraries(${sample8}
    MetavisionSDK::base
    MetavisionSDK::cv
    MetavisionSDK::cv3d
    MetavisionSDK::driver
    Boost::program_options
    ${CERES_LIBRARIES}
    Eigen3::Eigen
)

