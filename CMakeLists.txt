cmake_minimum_required(VERSION 3.16)
project(ZyboEdgeTPU LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(pybind11 REQUIRED)

add_subdirectory(src/cpp)