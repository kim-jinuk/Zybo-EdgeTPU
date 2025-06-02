file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS src/cpp/src/*.cpp)
file(GLOB_RECURSE HDR_FILES CONFIGURE_DEPENDS src/cpp/include/*.hpp)

add_library(zybo_edge SHARED ${SRC_FILES} ${HDR_FILES})

target_include_directories(zybo_edge PUBLIC src/cpp/include)

target_link_libraries(zybo_edge PUBLIC OpenCV::opencv pybind11::embed)