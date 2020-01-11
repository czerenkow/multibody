#pragma once
#include "common.h"

#ifdef ENABLE_VALIDATION_LAYERS

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData);

VkResult createDebugUtilsMessengerEXT(VkInstance instance,
                                      const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                      const VkAllocationCallbacks* pAllocator,
                                      VkDebugUtilsMessengerEXT* pDebugMessenger);
void destroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator);

//=========================================
// Setup and destroy DebugMessenger
//=========================================
VkDebugUtilsMessengerEXT setupDebugMessenger(VkInstance instance);
void destroyDebugMessager(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger);

#endif
