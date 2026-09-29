project(3d_pose)

cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)

find_package(MetavisionSDK COMPONENTS base cv cv3d driver REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)


set(sample metavision_active_marker_3d_test)
add_executable(${sample} ${sample}.cpp)
target_link_libraries(${sample}
    MetavisionSDK::base
    MetavisionSDK::cv
    MetavisionSDK::cv3d
    MetavisionSDK::driver
    Boost::program_options
)

set(sample2 metavision_active_marker_3d_pose)
add_executable(${sample2} ${sample2}.cpp)
target_link_libraries(${sample2}
    MetavisionSDK::base
    MetavisionSDK::cv
    MetavisionSDK::cv3d
    MetavisionSDK::driver
    Boost::program_options
)

set(sample3 pose_estimation_pnp)
add_executable(${sample3} ${sample3}.cpp)
target_link_libraries(${sample3}
    MetavisionSDK::base
    MetavisionSDK::cv
    MetavisionSDK::cv3d
    MetavisionSDK::driver
    Boost::program_options
)

