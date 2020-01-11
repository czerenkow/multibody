#pragma once
#include <vulkan/vulkan.hpp>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#ifndef NDEBUG
#define ENABLE_VALIDATION_LAYERS
#endif

#define ALLOCATOR nullptr

std::vector<const char*> to_c_pointers(const std::vector<std::string>& v);
