cmake_minimum_required(VERSION 3.18)
project(zybo_edge_tpu LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_library(EDGETPU edgetpu REQUIRED)
find_library(TFLITE tensorflow-lite REQUIRED PATHS /usr/local/lib)

include_directories(/usr/local/include)
link_directories(/usr/local/lib)

file(GLOB_RECURSE SRC
     CONFIGURE_DEPENDS
     src/*.cpp)

add_executable(app ${SRC})
target_include_directories(app PRIVATE include)
target_link_libraries(app PRIVATE ${OpenCV_LIBS} ${EDGETPU} ${TFLITE} pthread)
