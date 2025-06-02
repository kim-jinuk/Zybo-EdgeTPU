# ==== Zybo-EdgeTPU/src/cpp/CMakeLists.txt ====
# Build shared library `libzybo_edge.so`

file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE HDR_FILES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

add_library(zybo_edge SHARED ${SRC_FILES} ${HDR_FILES})

target_include_directories(zybo_edge PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(zybo_edge PUBLIC
    OpenCV::opencv
    Python3::Python
    pybind11::embed
)

# ----------------------------------------------------------------------------
#  * Build with:
#       mkdir build && cd build
#       cmake .. -DCMAKE_BUILD_TYPE=Release
#       make -j$(nproc)
#  * Cross‑compile:
#       source <petalinux>/environment-setup-aarch64-xilinx-linux
#       cmake .. -DCMAKE_BUILD_TYPE=Release \
#                -DCMAKE_C_COMPILER=${CROSS_COMPILE}gcc \
#                -DCMAKE_CXX_COMPILER=${CROSS_COMPILE}g++
# ----------------------------------------------------------------------------