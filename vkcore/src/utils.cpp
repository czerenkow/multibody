#include <vector>
#include <fstream>
#include "utils.hpp"
#include "vkcore.hpp"

std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }

    auto fileSize = file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}


VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    // TODO: I suppose that size of 'code' should be n*4
    createInfo.pCode = reinterpret_cast<const uint32_t*>( code.data() );
    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, ALLOCATOR, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shader module!");
    }
    return shaderModule;
}


VkShaderModule createShaderModule(VkDevice device, const std::string& filename) {
    return createShaderModule(device, readFile(filename));
}


// Buffer should be destroyed by:
// vkDestroyBuffer(device, vertexBuffer, nullptr);
VkBuffer createGPUBuffer(VkDevice device, VkDeviceSize size, VkBufferUsageFlags usage)
{
    VkBufferCreateInfo bufferInfo {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer result;
    if (vkCreateBuffer(device, &bufferInfo, ALLOCATOR, &result) != VK_SUCCESS) {
        throw std::runtime_error("createGPUBuffer");
    }
    return result;
}


VkDeviceMemory allocGPUMemory(VkDevice device,
                              VkDeviceSize allocationSize,
                              uint32_t memoryTypeIndex)
{
    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = allocationSize;
    allocInfo.memoryTypeIndex = memoryTypeIndex;

    VkDeviceMemory result;
    if (vkAllocateMemory(device, &allocInfo, ALLOCATOR, &result) != VK_SUCCESS) {
        throw std::runtime_error("allocGPUMemory");
    }
    return result;
}


VkMemoryRequirements getGPUBufferMemoryRequirements(VkDevice device, VkBuffer buffer)
{
    VkMemoryRequirements result;
    vkGetBufferMemoryRequirements(device, buffer, &result);
    return result;
}



Buffer::Buffer(VkDevice device,
       VkDeviceSize size,
       VkBufferUsageFlags usage,
       PhysicalDevice::amd_memory_type memType)
{
    std::cout << "Creating Buffer\n";
    buffer = createGPUBuffer(device, size, usage);
    VkMemoryRequirements memRequirements = getGPUBufferMemoryRequirements(device, buffer);
    // memoryTypeBits bit N is set - means that memory type index N is supported
    if ( ((1 << memType) & memRequirements.memoryTypeBits) == 0 ) {
        throw std::runtime_error("Buffer::Buffer improper memory type for this type of resource. Expected: " + std::to_string(memRequirements.memoryTypeBits));
    }
    memory = allocGPUMemory(device, memRequirements.size, memType);
    std::cout << "  Buffer: memory request: " << size << '\n';
    std::cout << "  Buffer:   memory alloc: " << memRequirements.size << '\n';

    // 4th param (0) is offset. If this is not eq 0, then it must be divisible by memRequirements.alignment
    if (vkBindBufferMemory(device, buffer, memory, 0) != VK_SUCCESS) {
        throw std::runtime_error("createBufferAndAlloc");
    }
    this->device = device;
}


Buffer::~Buffer() {
    if (!buffer) {
        return; // seems that object was moved
    }
    std::cout << "Destroying Buffer\n";
    vkDestroyBuffer(device, buffer, ALLOCATOR);
    vkFreeMemory(device, memory, ALLOCATOR);
}


std::string errorString(VkResult errorCode)
{
    switch (errorCode)
    {
#define STR(r) case VK_ ##r: return #r
        STR(NOT_READY);
        STR(TIMEOUT);
        STR(EVENT_SET);
        STR(EVENT_RESET);
        STR(INCOMPLETE);
        STR(ERROR_OUT_OF_HOST_MEMORY);
        STR(ERROR_OUT_OF_DEVICE_MEMORY);
        STR(ERROR_INITIALIZATION_FAILED);
        STR(ERROR_DEVICE_LOST);
        STR(ERROR_MEMORY_MAP_FAILED);
        STR(ERROR_LAYER_NOT_PRESENT);
        STR(ERROR_EXTENSION_NOT_PRESENT);
        STR(ERROR_FEATURE_NOT_PRESENT);
        STR(ERROR_INCOMPATIBLE_DRIVER);
        STR(ERROR_TOO_MANY_OBJECTS);
        STR(ERROR_FORMAT_NOT_SUPPORTED);
        STR(ERROR_SURFACE_LOST_KHR);
        STR(ERROR_NATIVE_WINDOW_IN_USE_KHR);
        STR(SUBOPTIMAL_KHR);
        STR(ERROR_OUT_OF_DATE_KHR);
        STR(ERROR_INCOMPATIBLE_DISPLAY_KHR);
        STR(ERROR_VALIDATION_FAILED_EXT);
        STR(ERROR_INVALID_SHADER_NV);
#undef STR
    default:
        return "UNKNOWN_ERROR";
    }
}



//======================================================================
// Synchronisation
//======================================================================


VkSemaphore createSemaphore(VkDevice device)
{
    VkSemaphoreCreateInfo semaphoreInfo;
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphoreInfo.pNext = nullptr;
    semaphoreInfo.flags = 0;

    VkSemaphore result;
    VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreInfo, ALLOCATOR, &result))
    return result;
}


VkFence createFenceSignaled(VkDevice device)
{
    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.pNext = nullptr;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    VkFence result;
    VK_CHECK_RESULT(vkCreateFence(device, &fenceInfo, ALLOCATOR, &result))
    return result;
}


/// Not signaled fence
VkFence createFence(VkDevice device)
{
    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.pNext = nullptr;
    fenceInfo.flags = 0; // Not signaled fence

    VkFence result;
    VK_CHECK_RESULT(vkCreateFence(device, &fenceInfo, ALLOCATOR, &result))
    return result;
}



//======================================================================
// Command Buffer
//======================================================================

std::vector<VkCommandBuffer> allocateCommandBuffers(VkDevice device, VkCommandPool commandPool, uint32_t numberOfBuffers)
{
    VkCommandBufferAllocateInfo allocInfo {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = numberOfBuffers;

    std::vector<VkCommandBuffer> commandBuffers(numberOfBuffers);
    VK_CHECK_RESULT( vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) )
    return commandBuffers;
}


VkCommandBuffer allocateCommandBuffer(VkDevice device, VkCommandPool commandPool)
{
    VkCommandBufferAllocateInfo allocInfo {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer))
    return commandBuffer;
}


void beginCommandBuffer_SingleTime(VkCommandBuffer cb)
{
    VkCommandBufferBeginInfo beginInfo {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK_RESULT(vkBeginCommandBuffer(cb, &beginInfo))
}


void beginCommandBuffer_SimultaneousUse(VkCommandBuffer cb)
{
    VkCommandBufferBeginInfo beginInfo {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    VK_CHECK_RESULT(vkBeginCommandBuffer(cb, &beginInfo))
}


void endCommandBuffer(VkCommandBuffer commandBuffer) {
    VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer))
}


void submitCommandBuffer(VkQueue queue,
                         VkCommandBuffer commandBuffer,
                         VkFence fence)
{
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    submitInfo.pWaitDstStageMask = nullptr;
    submitInfo.waitSemaphoreCount = 0;
    submitInfo.pWaitSemaphores = nullptr;
    submitInfo.signalSemaphoreCount = 0;
    submitInfo.pSignalSemaphores = nullptr;

    VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, fence))
}


void copyBuffer(VkDevice device,
                VkQueue queue,
                VkCommandPool commandPool,
                VkBuffer srcBuffer,
                VkBuffer dstBuffer,
                VkDeviceSize size)
{
    VkCommandBuffer commandBuffer = allocateCommandBuffer(device, commandPool);
    beginCommandBuffer_SingleTime(commandBuffer);
    VkBufferCopy copyRegion {};
    copyRegion.srcOffset = 0; // Optional
    copyRegion.dstOffset = 0; // Optional
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
    endCommandBuffer(commandBuffer);
    auto fence = createFence(device);
    submitCommandBuffer(queue, commandBuffer, fence);
    vkWaitForFences(device, 1, &fence, VK_TRUE, std::numeric_limits<uint64_t>::max());
    vkDestroyFence(device, fence, ALLOCATOR);
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}


//===========================================================================================

void transitionDepthImageToOptimal(VkDevice device,
                                   VkCommandPool commandPool,
                                   VkQueue queue,
                                   VkImage image,
                                   VkFormat format)
{
    VkCommandBuffer commandBuffer = allocateCommandBuffer(device, commandPool);
    beginCommandBuffer_SingleTime(commandBuffer);

    VkImageMemoryBarrier barrier {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    if (hasStencilComponent(format)) {
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    } else {
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    }

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );

    endCommandBuffer(commandBuffer);

    auto fence = createFence(device);
    submitCommandBuffer(queue, commandBuffer, fence);
    vkWaitForFences(device, 1, &fence, VK_TRUE, std::numeric_limits<uint64_t>::max());
    vkDestroyFence(device, fence, ALLOCATOR);
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}



//=============================
// Depth buffer
//=============================


Image createDepthImage(VkDevice device,
                       VkExtent3D extent,
                       VkFormat format)
{
    //----------------
    // Create image
    //----------------
    VkImageCreateInfo imageInfo {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent = extent;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.flags = 0;

    Image result;
    VK_CHECK_RESULT(vkCreateImage(device, &imageInfo, ALLOCATOR, &result.image))

    //--------------------------------
    // Allocate memory and bind
    //--------------------------------
    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, result.image, &memRequirements);

    VkMemoryAllocateInfo allocInfo {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = PhysicalDevice::amd_memory_type::device_local_primary_0;

    if (vkAllocateMemory(device, &allocInfo, ALLOCATOR, &result.memory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate depth image memory!");
    }

    vkBindImageMemory(device, result.image, result.memory, 0);

    //--------------------------------
    // Image view
    //--------------------------------
    VkImageViewCreateInfo viewInfo {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = result.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = imageInfo.format;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    VK_CHECK_RESULT(vkCreateImageView(device, &viewInfo, ALLOCATOR, &result.imageView))
    return result;
}

