#pragma once
#include <string>
#include "common.h"
#include "physical_device.h"
std::vector<char> readFile(const std::string& filename);

VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code);
VkShaderModule createShaderModule(VkDevice device, const std::string& filename);


VkBuffer createGPUBuffer(VkDevice device,
                         VkDeviceSize size,
                         VkBufferUsageFlags usage);
VkDeviceMemory allocGPUMemory(VkDevice device,
                              VkDeviceSize allocationSize,
                              uint32_t memoryTypeIndex);
VkMemoryRequirements getGPUBufferMemoryRequirements(VkDevice device, VkBuffer buffer);


struct Buffer {
    VkBuffer buffer;
    VkDeviceMemory memory;

    Buffer()
    {
        device = nullptr;
        buffer = nullptr;
        memory = nullptr;
    }

    Buffer(Buffer&& buffer):
        buffer{buffer.buffer},
        memory{buffer.memory},
        device{buffer.device}
    {
        buffer.buffer = nullptr;
        buffer.memory = nullptr;
        buffer.device = nullptr;
    }

    void operator=(Buffer&& buffer)
    {
        this->buffer = buffer.buffer;
        this->memory = buffer.memory;
        this->device = buffer.device;
        buffer.buffer = nullptr;
        buffer.memory = nullptr;
        buffer.device = nullptr;
    }

    Buffer(VkDevice device,
           PhysicalDevice& physicalDevice,
           VkDeviceSize size,
           VkBufferUsageFlags usage,
           VkMemoryPropertyFlags memoryFlags);
    ~Buffer();
private:
    VkDevice device; // only need by destructor
};



struct Image {
    VkImage image {VK_NULL_HANDLE};
    VkImageView imageView {VK_NULL_HANDLE};
    VkDeviceMemory memory {VK_NULL_HANDLE};

    void destroy(VkDevice device) {
        vkDestroyImage(device, image, ALLOCATOR);
        vkDestroyImageView(device, imageView, ALLOCATOR);
        vkFreeMemory(device, memory, ALLOCATOR);
    }
};


//=========================================================================================================
// Stollen from https://github.com/SaschaWillems/Vulkan
//=========================================================================================================
#define VK_CHECK_RESULT(f)																				\
{																										\
    VkResult res = (f);																					\
    if (res != VK_SUCCESS)																				\
    {																									\
        std::cout << "Fatal : VkResult is \"" << errorString(res) << "\" in " << __FILE__ << " at line " << __LINE__ << std::endl; \
        assert(0);																					    \
    }																									\
}

std::string errorString(VkResult errorCode);

//======================================================================

VkSemaphore createSemaphore(VkDevice device);
VkFence createFenceSignaled(VkDevice device);

/// Not signaled fence
VkFence createFence(VkDevice device);

std::vector<VkCommandBuffer> allocateCommandBuffers(VkDevice device, VkCommandPool commandPool, uint32_t numberOfBuffers);
VkCommandBuffer allocateCommandBuffer(VkDevice device, VkCommandPool commandPool);

void beginCommandBuffer_SingleTime(VkCommandBuffer cb);
void beginCommandBuffer_SimultaneousUse(VkCommandBuffer cb);
void endCommandBuffer(VkCommandBuffer commandBuffer);

void submitCommandBuffer(VkQueue queue,
                         VkCommandBuffer commandBuffer,
                         VkFence fence = VK_NULL_HANDLE);

void copyBuffer(VkDevice device,
                VkQueue queue,
                VkCommandPool commandPool,
                VkBuffer srcBuffer,
                VkBuffer dstBuffer,
                VkDeviceSize size);

void transitionDepthImageToOptimal(VkDevice device,
                                   VkCommandPool commandPool,
                                   VkQueue queue,
                                   VkImage image,
                                   VkFormat format);

Image createDepthImage(VkDevice device,
                       PhysicalDevice &physicalDevice,
                       VkExtent3D extent,
                       VkFormat format);
