#pragma once
#include "common.h"
#include <vulkan/vulkan.hpp>
#include <optional>
#include <set>
#include <iostream>


struct QueueFamilyIndices
{
    uint32_t graphicsComputeFamily;
    uint32_t presentFamily;

    std::set<uint32_t> uniqueIndices() {
        std::set<uint32_t> res;
        res.insert(graphicsComputeFamily);
        res.insert(presentFamily);
        return res;
    }
};


struct SwapChainSupportDetails
{
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};


class PhysicalDevice {
public:
    PhysicalDevice() {}
    PhysicalDevice(VkPhysicalDevice pdphysicalDevice);
    VkDevice createDevice(const std::vector<std::string>& deviceExtensions,
                          const std::set<uint32_t>& queueFamilyIndices);

    uint32_t findMemoryType(uint32_t memoryTypeBitsRequirement, VkMemoryPropertyFlags requiredProperties) const;
    void displaySubgroupInfo() const;

    QueueFamilyIndices findQueueFamilyIndices(VkSurfaceKHR surface);
    SwapChainSupportDetails querySwapChainSupport(VkSurfaceKHR surface);
    VkFormat findDepthFormat();

    bool checkPhysicalDeviceExtensionSupport(const std::vector<std::string>& extensions);
    bool checkPhysicalDeviceQueueFimiliesSupport();

    // Radeon RX 570:
    //    enum amd_memory_type {
    //        device_local_primary_0 = 0,
    //        host_local_primary_1,
    //        device_local_256MB_2,
    //        host_local_cached_3
    //    };
private:
    VkSampleCountFlagBits queryMaxUsableSampleCount();
    std::vector<VkQueueFamilyProperties> retrieveQueueFamilyProperties();
    std::optional<uint32_t> findGraphicsAndComputeQueueFamilyIndex();
    std::optional<uint32_t> findPresentQueueFamilyIndex(VkSurfaceKHR surface);
    VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);

private:
    VkPhysicalDevice physicalDevice;
    VkPhysicalDeviceMemoryProperties memoryProperties;

public:
     // TODO: move these to 'private'
    VkPhysicalDeviceProperties physicalDeviceProperties;
    VkSampleCountFlagBits maxUsableSampleCount;
    std::vector<VkQueueFamilyProperties> queueFamilyProperties;

};


bool hasStencilComponent(VkFormat format);





