#pragma once
#include "vkcore.hpp"


//===========================================
// Surface & SwapChain
//===========================================

struct SwapChain
{
    VkSwapchainKHR swapChain;
    VkFormat imageFormat;
    VkExtent2D extent;
    std::vector<VkImageView> imageViews;

    void destroy(VkDevice device) {
        for (auto iv: imageViews) {
            vkDestroyImageView(device, iv, ALLOCATOR);
        }
        vkDestroySwapchainKHR(device, swapChain, ALLOCATOR);
    }
};


SwapChain createSwapChain(VkDevice device,
                          PhysicalDevice& physicalDevice,
                          VkSurfaceKHR surface,
                          QueueFamilyIndices queueFamilyIndices,
                          VkExtent2D window_size);

//===========================================
// Drawing
//===========================================

uint32_t acquireNextImageFromSwapchain(VkDevice device, VkSwapchainKHR swapChain, VkSemaphore semaphore);
void presentFrame(VkQueue presentQueue,
                  VkSwapchainKHR swapChain,
                  uint32_t swapChainImageIndex,
                  VkSemaphore renderFinishedSemaphore);
