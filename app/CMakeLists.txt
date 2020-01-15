cmake_minimum_required(VERSION 3.16)

project(multibody LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(multibody)
target_sources(multibody
    PRIVATE
    "main.cpp"
    "compute.cpp"
    "graphics.cpp"
)
target_link_libraries(multibody PRIVATE vkcore)
target_link_libraries(multibody PRIVATE LibGLFW)

add_custom_target(Shaders ALL make -C ${CMAKE_SOURCE_DIR}/shaders/)
add_compile_definitions(SHADERS_DIR="${CMAKE_SOURCE_DIR}/shaders/")
