cmake_minimum_required(VERSION 3.10)

project(example-icp)
set(CMAKE_CXX_STANDARD 14)

add_executable(client client.cpp)

# if want to use off-the-shelf .so or .a
# target_link_libraries(hello ${CMAKE_SOURCE_DIR}/libgreet.so)
