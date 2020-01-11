#pragma once
#include <string>
#include "common.h"
#include "physical_device.h"
#include "debug.h"
#include "swap_chain.h"
#include "utils.hpp"

//namespace vkcore {
bool checkInstanceExtensionSupport(const std::vector<std::string>& extensions);

VkInstance createVulkanInstance(const std::vector<std::string>& instanceExtensions,
                                const std::vector<std::string>& validationLayers);
PhysicalDevice pickPhysicalDevice(VkInstance instance, const std::vector<std::string>& requiredDeviceExtensions);

VkCommandPool createCommandPool(VkDevice device, uint32_t queueFamilyIndex);

void displayVulkanVersionSupport();

//} // namespace: vkcore
