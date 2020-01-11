#include "physical_device.h"




PhysicalDevice::PhysicalDevice(VkPhysicalDevice physicalDevice): physicalDevice {physicalDevice} {
    vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);
    maxUsableSampleCount = VK_SAMPLE_COUNT_1_BIT; // 1 BIT is tmp, should be queryMaxUsableSampleCount();
    queueFamilyProperties = retrieveQueueFamilyProperties();

    auto version = physicalDeviceProperties.apiVersion;
    std::cout << "API version: "
        << VK_VERSION_MAJOR(version) << '.'
        << VK_VERSION_MINOR(version) << '.'
        << VK_VERSION_PATCH(version) << '\n';
    version = physicalDeviceProperties.driverVersion;
    std::cout << "Driver version: "
        << VK_VERSION_MAJOR(version) << '.'
        << VK_VERSION_MINOR(version) << '.'
        << VK_VERSION_PATCH(version) << '\n';

    // get memory types and heaps
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
}


VkSampleCountFlagBits PhysicalDevice::queryMaxUsableSampleCount() {
    VkSampleCountFlags counts = std::min(physicalDeviceProperties.limits.framebufferColorSampleCounts,
                                         physicalDeviceProperties.limits.framebufferDepthSampleCounts);
    if (counts & VK_SAMPLE_COUNT_64_BIT) { return VK_SAMPLE_COUNT_64_BIT; }
    if (counts & VK_SAMPLE_COUNT_32_BIT) { return VK_SAMPLE_COUNT_32_BIT; }
    if (counts & VK_SAMPLE_COUNT_16_BIT) { return VK_SAMPLE_COUNT_16_BIT; }
    if (counts & VK_SAMPLE_COUNT_8_BIT) { return VK_SAMPLE_COUNT_8_BIT; }
    if (counts & VK_SAMPLE_COUNT_4_BIT) { return VK_SAMPLE_COUNT_4_BIT; }
    if (counts & VK_SAMPLE_COUNT_2_BIT) { return VK_SAMPLE_COUNT_2_BIT; }

    return VK_SAMPLE_COUNT_1_BIT;
}



std::vector<VkQueueFamilyProperties> PhysicalDevice::retrieveQueueFamilyProperties() {
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

    return queueFamilies;
}


std::optional<uint32_t> PhysicalDevice::findGraphicsAndComputeQueueFamilyIndex() {
    uint32_t i = 0;
    for (const auto& queueFamily : queueFamilyProperties)
    {
        if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT &&
            queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT)
        {
            return i;
        }
        i++;
    }
    return {};
}


std::optional<uint32_t> PhysicalDevice::findPresentQueueFamilyIndex(VkSurfaceKHR surface) {
    uint32_t i = 0;
    for (const auto& queueFamily : queueFamilyProperties)
    {
        VkBool32 b;
        vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &b);
        if (b)
            return i;
        i++;
    }
    return {};
}


QueueFamilyIndices PhysicalDevice::findQueueFamilyIndices(VkSurfaceKHR surface)
{
    auto graphicsComputeFamily = findGraphicsAndComputeQueueFamilyIndex();
    if ( !graphicsComputeFamily.has_value() ) {
        throw std::runtime_error("failed to find graphicsComputeFamily!");
    }

    auto presentFamily = findPresentQueueFamilyIndex(surface);
    if ( !presentFamily.has_value() ) {
        throw std::runtime_error("failed to find presentFamily!");
    }

    return {graphicsComputeFamily.value(), presentFamily.value()};
}


bool PhysicalDevice::checkPhysicalDeviceExtensionSupport(const std::vector<std::string>& extensions){
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, availableExtensions.data());

    std::set<std::string> requiredDeviceExtensionsSet(extensions.begin(), extensions.end());

    for (const auto& extension : availableExtensions) {
        requiredDeviceExtensionsSet.erase(extension.extensionName);
    }

    return requiredDeviceExtensionsSet.empty();
}


SwapChainSupportDetails PhysicalDevice::querySwapChainSupport(VkSurfaceKHR surface) {
    SwapChainSupportDetails details;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &details.capabilities);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, nullptr);
    if (formatCount != 0)
    {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, details.formats.data());
    }

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, nullptr);
    if (presentModeCount != 0)
    {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, details.presentModes.data());
    }

    return details;
}


VkFormat PhysicalDevice::findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
    for (VkFormat format : candidates) {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

        if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
            return format;
        } else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
            return format;
        }
    }

    throw std::runtime_error("failed to find supported format!");
}

VkFormat PhysicalDevice::findDepthFormat() {
    return findSupportedFormat(
        {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
    );
}


// XXX This is removed as I suppose that this is not the best way to find correct memory type
//uint32_t PhysicalDevice::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
//{
//    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
//        if ( (typeFilter & (1 << i)) && // bit i == true, means that memory type i is supported
//             (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
//        {
//            return i;
//        }
//    }
//    throw std::runtime_error("failed to find suitable memory type!");
//}


bool PhysicalDevice::checkPhysicalDeviceQueueFimiliesSupport() {
    std::cout << "Not implemented: checkPhysicalDeviceQueueFimiliesSupport\n";
    return true;
}



VkDevice PhysicalDevice::createDevice(const std::vector<std::string>& deviceExtensions,
                                      const std::set<uint32_t>& queueFamilyIndices)
{
    float queuePriority = 1.0f;
    // TODO: this seems to be bad idea
    auto queueCreateInfos = [&queuePriority](std::set<uint32_t> uniqueQueueFamilyIndices) -> std::vector<VkDeviceQueueCreateInfo>
    {
        std::vector<VkDeviceQueueCreateInfo> result;
        for (uint32_t queueFamily : uniqueQueueFamilyIndices) {
            VkDeviceQueueCreateInfo queueCreateInfo {};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1; // there are more queue (I see that it can be e.g. 4) but we want to creat only one
            assert(queueCreateInfo.queueCount == 1); //
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfo.flags = 0;

            result.push_back(queueCreateInfo);
        }
        return result;
    };
    auto queueCreateInfosData = queueCreateInfos( queueFamilyIndices);

    // TODO: this needs to be checked if particular fature is available
    VkPhysicalDeviceFeatures deviceFeatures {}; // for now we do not require any special features
    //deviceFeatures.samplerAnisotropy = VK_FALSE;
    //deviceFeatures.shaderFloat64 = VK_TRUE;
    // fillModeNonSolid specifies whether point and wireframe fill modes are supported.
    // If this feature is not enabled, the VK_POLYGON_MODE_POINT and VK_POLYGON_MODE_LINE enum values must not be used.
    //deviceFeatures.fillModeNonSolid = VK_TRUE;
    //deviceFeatures.logicOp = VK_TRUE;
    //deviceFeatures.dualSrcBlend = VK_TRUE;

    const auto deviceExtensions_p = to_c_pointers(deviceExtensions);

    VkDeviceCreateInfo createInfo {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.queueCreateInfoCount = static_cast<uint32_t>( queueCreateInfosData.size() );
    createInfo.pQueueCreateInfos = queueCreateInfosData.data();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions_p.size()); // We do not need any extension to render offline
    createInfo.ppEnabledExtensionNames = deviceExtensions_p.data();
    createInfo.enabledLayerCount = 1; // DEPRECATED - ignored
    const char *xx = {"VK_LAYER_KHRONOS_validation"};
    createInfo.ppEnabledLayerNames = &xx; // DEPRECATED - ignored

    VkDevice device;
    if (vkCreateDevice(physicalDevice, &createInfo, ALLOCATOR, &device) != VK_SUCCESS)
        throw std::runtime_error("failed to create logical device!");

    return device;
}


void PhysicalDevice::displaySubgroupInfo() {
    VkPhysicalDeviceSubgroupProperties subgroupProperties;
    subgroupProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    subgroupProperties.pNext = nullptr;

    VkPhysicalDeviceProperties2 physicalDeviceProperties;
    physicalDeviceProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    physicalDeviceProperties.pNext = &subgroupProperties;

    vkGetPhysicalDeviceProperties2(physicalDevice, &physicalDeviceProperties);

    std::cout << "Subgroup Properties\n";
    std::cout << "         subgroupSize: " <<  subgroupProperties.subgroupSize  << '\n';
    std::cout << "  supportedOperations: " <<  subgroupProperties.supportedOperations  << '\n';
}





bool hasStencilComponent(VkFormat format) {
    return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}
