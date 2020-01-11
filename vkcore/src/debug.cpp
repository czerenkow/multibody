#include "debug.h"
#include <iostream>

#ifdef ENABLE_VALIDATION_LAYERS

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData)
{
    std::cerr << "VL: " << pCallbackData->pMessage << std::endl;
    return VK_FALSE;
}


VkResult createDebugUtilsMessengerEXT(VkInstance instance,
                                      const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                      const VkAllocationCallbacks* pAllocator,
                                      VkDebugUtilsMessengerEXT* pDebugMessenger)
{
    auto func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>( vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT") );
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    }
    return VK_ERROR_EXTENSION_NOT_PRESENT;
}


void destroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>( vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT") );
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

//=========================================
// Setup and destroy DebugMessenger
//=========================================

VkDebugUtilsMessengerEXT setupDebugMessenger(VkInstance instance) {
    VkDebugUtilsMessengerCreateInfoEXT createInfo {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;
    createInfo.pUserData = nullptr;

    VkDebugUtilsMessengerEXT debugMessenger;
    if (createDebugUtilsMessengerEXT(instance, &createInfo, ALLOCATOR, &debugMessenger) != VK_SUCCESS)
        throw std::runtime_error("failed to set up debug messenger!");
    return debugMessenger;
}


void destroyDebugMessager(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger) {
    destroyDebugUtilsMessengerEXT(instance, debugMessenger, ALLOCATOR);
}

#endif
