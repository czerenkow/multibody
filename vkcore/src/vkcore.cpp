#include "vkcore.hpp"
#include "debug.h"
#include <string>

//=========================================
// Extensions / Layers
//=========================================


bool checkInstanceExtensionSupport(const std::vector<std::string>& extensions)
{
    // TODO: implement
    std::cout << "Not implemented: checkInstanceRequiredExtensionSupport\n";
    return true;
}



VkInstance createVulkanInstance(const std::vector<std::string>& instanceExtensions,
                                const std::vector<std::string>& validationLayers)
{
    VkApplicationInfo appInfo {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Triangle";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_1;  // TODO: if want to have 1.1 support -> VK_API_VERSION_1_1

    VkInstanceCreateInfo createInfo {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    checkInstanceExtensionSupport(instanceExtensions);

    const auto instanceExtensions_p = to_c_pointers(instanceExtensions);
    createInfo.enabledExtensionCount = static_cast<uint32_t>(instanceExtensions_p.size());
    createInfo.ppEnabledExtensionNames = instanceExtensions_p.data();
    const auto validationLayers_p = to_c_pointers(validationLayers);
    createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers_p.size());
    createInfo.ppEnabledLayerNames = validationLayers_p.data();

    VkInstance instance;
    if (vkCreateInstance(&createInfo, ALLOCATOR, &instance) != VK_SUCCESS)
        throw std::runtime_error("failed to create instance!");
    return instance;
}


PhysicalDevice pickPhysicalDevice(VkInstance instance, const std::vector<std::string>& requiredDeviceExtensions) {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

    if (deviceCount == 0)
        throw std::runtime_error("failed to find GPUs with Vulkan support!");

    std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, physicalDevices.data());

    // 1. Check if PhysicalDevice supports all required queue families
    // 2. Check if PhysicalDevice supports all required device extensions
    std::cout << "Number of physical devices: " << deviceCount << '\n';

    for (auto device: physicalDevices) {
        PhysicalDevice physicalDevice {device};
        if ( physicalDevice.checkPhysicalDeviceExtensionSupport(requiredDeviceExtensions) &&
             physicalDevice.checkPhysicalDeviceQueueFimiliesSupport() ) {
            return physicalDevice;
        }
    }
    throw std::runtime_error("failed to find sitable GPU!");
}



VkCommandPool createCommandPool(VkDevice device, uint32_t queueFamilyIndex) {
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamilyIndex;
    poolInfo.flags = 0; // Optional

    VkCommandPool commandPool;
    if (vkCreateCommandPool(device, &poolInfo, ALLOCATOR, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create command pool!");
    }
    return commandPool;
}

//=============================
// Utils
//=============================

void displayVulkanVersionSupport() {
    uint32_t apiVersion = 0;
    vkEnumerateInstanceVersion(&apiVersion);
    std::cout << "Max Instance API version that can be created: " << apiVersion << '\n';
    if (VK_MAKE_VERSION(1, 1, 0) <= apiVersion) {
        std::cout << "1.1 or newer is available\n";
    }
}
