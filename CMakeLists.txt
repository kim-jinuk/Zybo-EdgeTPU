# C++ 모듈이 필요 없으면 삭제 가능
cmake_minimum_required(VERSION 3.14)
project(zybo_eo_cpp LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
add_subdirectory(src/cpp)