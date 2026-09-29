cmake_minimum_required(VERSION 3.5)
project(pose_pangolin)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

# MetavisionSDK
find_package(MetavisionSDK COMPONENTS base cv cv3d driver REQUIRED)

# Boost
find_package(Boost COMPONENTS program_options REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED)

# OpenGL & GLEW
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

# Pangolin
find_package(Pangolin REQUIRED)


set(sample pose_pangolin)
add_executable(${sample} ${sample}.cpp)

target_link_libraries(${sample}
    MetavisionSDK::base
    MetavisionSDK::cv
    MetavisionSDK::cv3d
    MetavisionSDK::driver
    Boost::program_options
    ${OpenCV_LIBS}
    ${OPENGL_LIBRARIES}
    GLEW::GLEW

    # Pangolin
    pango_core
    pango_opengl
    pango_windowing
    pango_scene
    pango_plot
)


set(sample2 6dof)
add_executable(${sample2} ${sample2}.cpp)

target_link_libraries(${sample2}
    MetavisionSDK::base
    MetavisionSDK::cv
    MetavisionSDK::cv3d
    MetavisionSDK::driver
    Boost::program_options
    ${OpenCV_LIBS}
    ${OPENGL_LIBRARIES}
    GLEW::GLEW

    # Pangolin
    pango_core
    pango_opengl
    pango_windowing
    pango_scene
    pango_plot
)


set(sample3 visualize_6dof)
add_executable(${sample3} ${sample3}.cpp)

target_link_libraries(${sample3}
    MetavisionSDK::base
    MetavisionSDK::cv
    MetavisionSDK::cv3d
    MetavisionSDK::driver
    Boost::program_options
    ${OpenCV_LIBS}
    ${OPENGL_LIBRARIES}
    GLEW::GLEW

    # Pangolin
    pango_core
    pango_opengl
    pango_windowing
    pango_scene
    pango_plot
)


set(sample4 visualize_KF)
add_executable(${sample4} ${sample4}.cpp)

target_link_libraries(${sample4}
    MetavisionSDK::base
    MetavisionSDK::cv
    MetavisionSDK::cv3d
    MetavisionSDK::driver
    Boost::program_options
    ${OpenCV_LIBS}
    ${OPENGL_LIBRARIES}
    GLEW::GLEW

    # Pangolin
    pango_core
    pango_opengl
    pango_windowing
    pango_scene
    pango_plot
)


set(sample5 visualize_KF_v2)
add_executable(${sample5} ${sample5}.cpp)

target_link_libraries(${sample5}
    MetavisionSDK::base
    MetavisionSDK::cv
    MetavisionSDK::cv3d
    MetavisionSDK::driver
    Boost::program_options
    ${OpenCV_LIBS}
    ${OPENGL_LIBRARIES}
    GLEW::GLEW

    # Pangolin
    pango_core
    pango_opengl
    pango_windowing
    pango_scene
    pango_plot
)


set(sample6 visualize_RANSAC)
add_executable(${sample6} ${sample6}.cpp)

target_link_libraries(${sample6}
    MetavisionSDK::base
    MetavisionSDK::cv
    MetavisionSDK::cv3d
    MetavisionSDK::driver
    Boost::program_options
    ${OpenCV_LIBS}
    ${OPENGL_LIBRARIES}
    GLEW::GLEW

    # Pangolin
    pango_core
    pango_opengl
    pango_windowing
    pango_scene
    pango_plot
)
