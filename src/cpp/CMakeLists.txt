cmake_minimum_required(VERSION 3.15)
project(edge_tpu_pipeline)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# -------------------------
# 1) 먼저 시스템에 설치된 OpenCV 를 찾는다.
#    (apt install libopencv-dev 등)
# -------------------------
find_package(OpenCV QUIET PATHS
    /usr/local/lib/cmake/opencv4
    /usr/lib/x86_64-linux-gnu/cmake/opencv4)

# -------------------------
# 2) 못 찾으면 → FetchContent 로 opencv + opencv_contrib 를 소스부터 빌드한다.
# -------------------------
if(NOT OpenCV_FOUND)
  include(FetchContent)
  set(OPENCV_VERSION 4.9.0)         # 원하는 버전
  FetchContent_Declare(opencv
    GIT_REPOSITORY https://github.com/opencv/opencv.git
    GIT_TAG        ${OPENCV_VERSION})
  FetchContent_Declare(opencv_contrib
    GIT_REPOSITORY https://github.com/opencv/opencv_contrib.git
    GIT_TAG        ${OPENCV_VERSION})

  # OpenCV 옵션 최소화 – Python 바인딩/테스트 OFF
  set(BUILD_opencv_python OFF CACHE BOOL "" FORCE)
  set(BUILD_TESTS         OFF CACHE BOOL "" FORCE)
  set(BUILD_DOCS          OFF CACHE BOOL "" FORCE)
  set(OPENCV_ENABLE_NONFREE OFF CACHE BOOL "" FORCE)
  set(OPENCV_EXTRA_MODULES_PATH ${opencv_contrib_SOURCE_DIR}/modules CACHE PATH "" FORCE)
  FetchContent_MakeAvailable(opencv opencv_contrib)
  # FetchContent 가 `opencv` 타깃(opencv::opencv) 을 내보냄
endif()

# -------------------------
# Edge‑TPU / TFLite
# -------------------------
# 옵션: 자체 설치한 Coral 런타임 경로를 넘길 수 있게 함
option(EDGETPU_ROOT "Preset path to Coral Edge‑TPU runtime (contains lib/ + include/)" "")
if(EDGETPU_ROOT)
  list(APPEND CMAKE_PREFIX_PATH ${EDGETPU_ROOT})
endif()

# ---- Edge‑TPU runtime ----
find_library(EDGETPU_LIB NAMES edgetpu
             HINTS ${EDGETPU_ROOT}/lib
                   /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib
             REQUIRED)
find_path   (EDGETPU_INC edgetpu.h
             HINTS ${EDGETPU_ROOT}/include
                   /usr/include /usr/local/include
             REQUIRED)

# ---- TensorFlow‑Lite ----
find_library(TFLITE_LIB NAMES tensorflow-lite tensorflowlite PATHS
             /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)
find_path  (TFLITE_INC tensorflow/lite/interpreter.h PATHS /usr/include /usr/local/include)

# 없으면 FetchContent 로 minimal 빌드
if(NOT TFLITE_LIB OR NOT TFLITE_INC)
  include(FetchContent)
  FetchContent_Declare(tflite
    GIT_REPOSITORY https://github.com/tensorflow/tflite-support.git
    GIT_TAG v0.4.5)
  set(TFLITE_ENABLE_XNNPACK OFF CACHE BOOL "" FORCE)
  set(TFLITE_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(tflite)
  set(TFLITE_LIB tensorflow-lite CACHE FILEPATH "")
  set(TFLITE_INC ${tflite_SOURCE_DIR} CACHE PATH "")
endif()

include_directories(
    include
    ${TFLITE_INC}
    ${EDGETPU_INC}
)

add_subdirectory(src)