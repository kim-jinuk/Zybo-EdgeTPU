file(GLOB_RECURSE SRC *.cpp)
add_executable(app ${SRC})
target_link_libraries(app ${OpenCV_LIBS} ${TFLITE_LIB} ${EDGETPU_LIB} pthread)