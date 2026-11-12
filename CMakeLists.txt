cmake_minimum_required(VERSION 3.10)
project(robokin VERSION 0.1.0 LANGUAGES CXX)

# The library targets C++98 deliberately: it must build on the older
# toolchains still common on industrial controllers and embedded targets.
set(CMAKE_CXX_STANDARD 98)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(robokin STATIC
  src/Matrix.cpp
  src/Kinematics.cpp
  src/Trajectory.cpp)

target_include_directories(robokin PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

if(MSVC)
  target_compile_options(robokin PRIVATE /W4)
else()
  target_compile_options(robokin PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---- demo ----
add_executable(puma560_demo examples/puma560_demo.cpp)
target_link_libraries(puma560_demo PRIVATE robokin)

# ---- tests ----
enable_testing()
add_executable(test_robo tests/test_robo.cpp)
target_link_libraries(test_robo PRIVATE robokin)
add_test(NAME unit COMMAND test_robo)
