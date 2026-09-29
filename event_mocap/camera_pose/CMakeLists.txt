project(camera_pose)

cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

find_package(MetavisionSDK COMPONENTS base cv cv3d driver REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})


set(sample camera_pose_csv)
add_executable(${sample} ${sample}.cpp)
target_link_libraries(${sample}
    MetavisionSDK::base
    MetavisionSDK::cv
    MetavisionSDK::cv3d
    MetavisionSDK::driver
    Boost::program_options
)


set(sample2 camera_pose_v2)
add_executable(${sample2} ${sample2}.cpp)
target_link_libraries(${sample2}
    MetavisionSDK::base
    MetavisionSDK::cv
    MetavisionSDK::cv3d
    MetavisionSDK::driver
    Boost::program_options
)


set(sample3 camera_pose_bundle)
add_executable(${sample3} ${sample3}.cpp)
target_link_libraries(${sample3}
    MetavisionSDK::base
    MetavisionSDK::cv
    MetavisionSDK::cv3d
    MetavisionSDK::driver
    Boost::program_options
    ${CERES_LIBRARIES}
    Eigen3::Eigen
)


set(sample4 camera_pose_IPPE)
add_executable(${sample4} ${sample4}.cpp)
target_link_libraries(${sample4}
    MetavisionSDK::base
    MetavisionSDK::cv
    MetavisionSDK::cv3d
    MetavisionSDK::driver
    Boost::program_options
    ${CERES_LIBRARIES}
    Eigen3::Eigen
)
