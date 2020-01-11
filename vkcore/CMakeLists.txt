cmake_minimum_required(VERSION 3.16)

project(vkcore_proj LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(vkcore STATIC)
target_sources(vkcore 
	PRIVATE
	"src/common.cpp"
	"src/debug.cpp"
	"src/physical_device.cpp"
	"src/swap_chain.cpp"
	"src/utils.cpp"
	"src/vkcore.cpp"
	)
target_include_directories(vkcore PUBLIC "include")
target_link_libraries(vkcore PUBLIC Vulkan::Vulkan)
