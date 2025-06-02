# ------------------------------- CMake build system -----------------------------
# Two‑level hierarchy:
#  1. **Root CMakeLists.txt** – project settings, dependencies, demo target, and
#     `add_subdirectory(src/cpp)` so the library lives in its own folder.
#  2. **src/cpp/CMakeLists.txt** – actually collects all *.cpp/*.hpp under
#     `src/cpp/src` and `src/cpp/include`, builds `libzybo_edge.so`, links
#     OpenCV + pybind11 + Python.
# -------------------------------------------------------------------------------

# ==== Zybo-EdgeTPU/CMakeLists.txt (root) ====
cmake_minimum_required(VERSION 3.16)
project(ZyboEdgeTPU LANGUAGES C CXX)

option(BUILD_DEMO "Build demo main" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------- Dependencies ----------
find_package(OpenCV REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Python3 COMPONENTS Development REQUIRED)

# ---------- Library sub‑directory ----------
add_subdirectory(src/cpp)

# ---------- Optional demo executable ----------
if(BUILD_DEMO)
    add_executable(demo_main src/cpp/main.cpp)
    target_link_libraries(demo_main PRIVATE zybo_edge)
endif()
