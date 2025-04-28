add_library(frame_tools
    include/frame_tools/frame_draw.hpp
    src/frame_draw.cpp)
find_package(OpenCV REQUIRED)
set_target_properties(frame_tools PROPERTIES POSITION_INDEPENDENT_CODE ON)
include_directories(include)
target_link_libraries(frame_tools PUBLIC ${OpenCV_LIBS})