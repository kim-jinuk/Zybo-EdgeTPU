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
  set(OPENCV_VERSION 4.9.0)

  # 1. 두 레포 선언
  FetchContent_Declare(opencv
    GIT_REPOSITORY https://github.com/opencv/opencv.git
    GIT_TAG        ${OPENCV_VERSION})
  FetchContent_Declare(opencv_contrib
    GIT_REPOSITORY https://github.com/opencv/opencv_contrib.git
    GIT_TAG        ${OPENCV_VERSION})

  # 2. ***소스만*** 먼저 내려받아 위치를 알아낸다
  FetchContent_Populate(opencv_contrib)

  # 3. 이제 정확한 경로가 생겼으니 캐시에 넣기
  set(OPENCV_EXTRA_MODULES_PATH
      "${opencv_contrib_SOURCE_DIR}/modules" CACHE PATH "" FORCE)

  # 4. OpenCV & contrib 을 실제로 configure / build
  FetchContent_MakeAvailable(opencv opencv_contrib)
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
                   /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib)
# ↳ libedgetpu.so 심볼릭 링크가 없을 경우 .so.1 등 버전 파일을 직접 잡아 주기
if(NOT EDGETPU_LIB)
  file(GLOB _TPU_LIBS "${EDGETPU_ROOT}/lib/libedgetpu.so.*")
  if(_TPU_LIBS)
    list(GET _TPU_LIBS 0 EDGETPU_LIB)   # 첫 번째 후보 사용
  endif()
endif()
# 마지막에도 없으면 오류
if(NOT EDGETPU_LIB)
  message(FATAL_ERROR "Edge‑TPU library (libedgetpu.so) not found in ${EDGETPU_ROOT}/lib; make sure symlink exists or pass correct EDGETPU_ROOT")
endif()
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

  # ★ TensorFlow 2.14 소스의 lite 디렉터리만 빌드
  FetchContent_Declare(tflite
    GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
    GIT_TAG        v2.14.0
    SOURCE_SUBDIR  tensorflow/lite)

  # 빌드 옵션 최소화
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(TFLITE_ENABLE_RUY       OFF CACHE BOOL "" FORCE)
  set(TFLITE_ENABLE_XNNPACK   OFF CACHE BOOL "" FORCE)
  set(TFLITE_ENABLE_NNAPI     OFF CACHE BOOL "" FORCE)

  FetchContent_MakeAvailable(tflite)

  # FetchContent 가 생성한 static 라이브러리 타깃을 사용
  set(TFLITE_LIB tensorflow-lite-static)   # ← CMake 가 자동으로 만듦
  set(TFLITE_INC ${tflite_SOURCE_DIR}      CACHE PATH "")
endif()

if(NOT IS_DIRECTORY ${TFLITE_INC})
  message(FATAL_ERROR "TFLite headers not found! Hint: apt install libtensorflow-lite-dev \
  or pass -DTFLITE_INC=/path/to/include -DTFLITE_LIB=/path/to/libtensorflow-lite.a")
endif()

include_directories(
    ${PROJECT_SOURCE_DIR}/include        # 우리 헤더
    ${TFLITE_INC}                        # <tensorflow/lite/...>
    ${EDGETPU_INC})                      # <edgetpu.h>

add_subdirectory(src)