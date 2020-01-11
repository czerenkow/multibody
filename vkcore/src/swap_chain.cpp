#include "swap_chain.h"

//===========================================
// namespace: private
//===========================================
namespace  {

VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
    // PW: This is from Intel Tutorial:
    // ---
    // If the list contains only one entry with undefined format
    // it means that there are no preferred surface formats and any can be chosen
    if( (availableFormats.size() == 1) &&
            availableFormats[0].format == VK_FORMAT_UNDEFINED ) {
        return{ VK_FORMAT_R8G8B8A8_UNORM, VK_COLORSPACE_SRGB_NONLINEAR_KHR };
    }
    // ---

    for (const auto& availableFormat : availableFormats) {
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM &&
                availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        {
            return availableFormat;
        }
    }
    // TODO: think about this second one: {format: B8G8R8A8_SRGB, colorSpace: VK_COLOR_SPACE_SRGB_NONLINEAR_KHR}
    return availableFormats[0]; // we know that it is there
}


VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR> availablePresentModes) 
{
    return VK_PRESENT_MODE_IMMEDIATE_KHR; // XXX: temporary to get 60 fps
    if (std::find(availablePresentModes.cbegin(),
                  availablePresentModes.cend(),
                  VK_PRESENT_MODE_MAILBOX_KHR) != availablePresentModes.cend())
    {
        return VK_PRESENT_MODE_MAILBOX_KHR;
    }

    /* In tutorial we have:
     * Unfortunately some drivers currently don't properly support VK_PRESENT_MODE_FIFO_KHR,
     * so we should prefer VK_PRESENT_MODE_IMMEDIATE_KHR if VK_PRESENT_MODE_MAILBOX_KHR is not available.
     */
    return VK_PRESENT_MODE_FIFO_KHR;
}


VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities, VkExtent2D windowActualExtent) 
{
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        // it means
        return capabilities.currentExtent;
    } else {
        // select that matches the best
        windowActualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, windowActualExtent.width));
        windowActualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, windowActualExtent.height));
        return windowActualExtent;
    }
}


std::vector<VkImageView> createSwapChainImageViews(VkDevice device, const std::vector<VkImage>& images, VkFormat format) 
{
    std::vector<VkImageView> result(images.size());
    for (size_t i = 0; i < images.size(); i++) {
        VkImageViewCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image = images[i];
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = format;
        createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;
        if (vkCreateImageView(device, &createInfo, ALLOCATOR, &result[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image views!");
        }
    }
    return result;
}


// TODO: It is called 2 times. Is it expensive?
std::vector<VkImage> getSwapchainImages(VkDevice device, VkSwapchainKHR swapchain) {
    uint32_t imageCount;
    vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr);
    std::vector<VkImage> images(imageCount);
    vkGetSwapchainImagesKHR(device, swapchain, &imageCount, images.data());
    return images;
}

} // namespace: private

//===========================================
// Surface & SwapChain
//===========================================

SwapChain createSwapChain(VkDevice device,
                          PhysicalDevice& physicalDevice,
                          VkSurfaceKHR surface,
                          QueueFamilyIndices queueFamilyIndices,
                          VkExtent2D window_size)
{
    SwapChainSupportDetails swapChainSupport = physicalDevice.querySwapChainSupport(surface);
    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
    VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities, window_size);

    // Tell how many images we would like to have in the swap chain.
    // Simply sticking to this minimum means that we may sometimes have to wait on the driver
    // to complete internal operations before we can acquire another image to render to.
    // Therefore it is recommended to request at least one more image than the minimum.
    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
        imageCount = swapChainSupport.capabilities.maxImageCount;
        // INFO: swapChainSupport.capabilities.maxImageCount == 0 means that there is no maximum
    }

    VkSwapchainCreateInfoKHR createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    createInfo.preTransform = swapChainSupport.capabilities.currentTransform; // We are not going to do any transforms as part of the presentation operation.
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = VK_NULL_HANDLE;
    // TODO: In Intel Tutorial:
    // In this example we define additional “transfer destination” usage which is required for image clear operation.
    // TODO: verify it as here we use clear operation but we do not have this flag specified

    // We need to tell which queue family indices having access to the images(s) of the swapchain
    // when imageSharingMode is VK_SHARING_MODE_CONCURRENT.
    uint32_t indices[] = {queueFamilyIndices.graphicsComputeFamily, queueFamilyIndices.presentFamily};
    if (queueFamilyIndices.graphicsComputeFamily != queueFamilyIndices.presentFamily) {
        // 2 queues have access:
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = indices;
    } else {
        // Only one queue has access so we do not need to specify indices:
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.queueFamilyIndexCount = 0; // Optional
        createInfo.pQueueFamilyIndices = nullptr; // Optional
    }

    VkSwapchainKHR swapChain;
    if (vkCreateSwapchainKHR(device, &createInfo, ALLOCATOR, &swapChain) != VK_SUCCESS) {
        throw std::runtime_error("failed to create swap chain!");
    }

    // generating result
    SwapChain result;
    result.swapChain = swapChain;

    // And remeber format and extent for future pusposes:
    result.imageFormat = surfaceFormat.format;
    result.extent = extent;
    auto images = getSwapchainImages(device, swapChain);
    result.imageViews = createSwapChainImageViews(device, images, result.imageFormat);
    std::cout << "Images in swapchain: " << result.imageViews.size() << '\n';

    return result;
}

//===========================================
// Drawing
//===========================================

// Acquiring an image from the swap chain, returns its index.
// Blocks until the image is acquired, but it is possible that presentation engine still reads image data.
// The semaphore is signaled when it is done.
uint32_t acquireNextImageFromSwapchain(VkDevice device, VkSwapchainKHR swapChain, VkSemaphore semaphore) {
    uint32_t imageIndex;
    auto result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, semaphore, VK_NULL_HANDLE, &imageIndex);
    // TODO: add support to VK_ERROR_OUT_OF_DATE_KHR and VK_SUBOPTIMAL_KHR
    if (result != VK_SUCCESS) {
        std::cout << "error? acquireNextImageFromSwapchain: " << result << '\n';
        throw std::runtime_error("vkAcquireNextImageKHR: returned not VK_SUCCESS");
    }
    return imageIndex;
}


void presentFrame(VkQueue presentQueue,
                  VkSwapchainKHR swapChain,
                  uint32_t swapChainImageIndex,
                  VkSemaphore renderFinishedSemaphore)
{
    VkSemaphore signalSemaphores[] = {renderFinishedSemaphore};
    VkSwapchainKHR swapChains[] = {swapChain};

    VkPresentInfoKHR presentInfo {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.pResults = nullptr; // Optional

    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &swapChainImageIndex;

    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;

    auto result = vkQueuePresentKHR(presentQueue, &presentInfo);
    if (result != VK_SUCCESS) {
        // TODO: from some reason on my platform it never happens
        throw std::runtime_error("vkQueuePresentKHR: wow, it has happed!");
    }
}
