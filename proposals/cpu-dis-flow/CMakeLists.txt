cmake_minimum_required(VERSION 3.20)
project(cpu_dis_pilot LANGUAGES CXX)
find_package(OpenCV 4.12.0 EXACT REQUIRED COMPONENTS core imgproc imgcodecs video)
add_executable(flow_bench benchmark.cpp flow.cpp)
target_compile_features(flow_bench PRIVATE cxx_std_17)
target_include_directories(flow_bench PRIVATE ${OpenCV_INCLUDE_DIRS})
# Allows distributions that split the optional video-module dependencies
# to link just the optical-flow objects from a matching OpenCV archive.
set(DIS_VIDEO_LIBRARY opencv_video CACHE STRING "OpenCV video library or matching archive")
target_link_libraries(flow_bench PRIVATE ${DIS_VIDEO_LIBRARY} opencv_imgcodecs opencv_imgproc opencv_core)

include(CTest)
if(BUILD_TESTING)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME masked_endpoint_error
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_scoring.py
                     $<TARGET_FILE:flow_bench>)
endif()
