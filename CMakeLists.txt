# ==== Zybo-EdgeTPU/CMakeLists.txt (root) ====
cmake_minimum_required(VERSION 3.16)
project(ZyboEdgeTPU LANGUAGES C CXX)

option(BUILD_DEMO "Build demo main" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------- Dependencies ----------
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui photo)
find_package(pybind11 REQUIRED)
# Python is pulled in via pybind11::embed automatically.
# (If you prefer explicit link, uncomment the next line and ensure python3-dev is installed.)
# find_package(Python3 COMPONENTS Development REQUIRED)

# ---------- Library sub‑directory ----------
add_subdirectory(src/cpp)

# ---------- Optional demo executable ----------
if(BUILD_DEMO)
    add_executable(demo_main src/cpp/main.cpp)
    target_link_libraries(demo_main PRIVATE zybo_edge ${OpenCV_LIBS} ffi)
endif()
