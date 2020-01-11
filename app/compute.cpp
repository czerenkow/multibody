#include "compute.h"
#include "utils.hpp"

namespace compute {

VkDescriptorSetLayout createDescriptorSetLayout(VkDevice device) {
    VkDescriptorSetLayoutBinding buffer_in_out {};
    buffer_in_out.binding = 0;
    buffer_in_out.descriptorCount = 1;
    buffer_in_out.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    buffer_in_out.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    buffer_in_out.pImmutableSamplers = nullptr; // Optional

    std::array<VkDescriptorSetLayoutBinding, 1> bindings = {buffer_in_out};

    VkDescriptorSetLayoutCreateInfo layoutInfo {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>( bindings.size() );
    layoutInfo.pBindings = bindings.data();

    VkDescriptorSetLayout result;
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &layoutInfo, ALLOCATOR, &result))
    return result;
}


VkPipelineLayout createPipelineLayout(VkDevice device, VkDescriptorSetLayout descriptorSetLayout)
{
//    VkPushConstantRange pc {};
//    pc.size = sizeof(uint32_t);
//    pc.offset = 0;
//    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges = nullptr; // &pc

    VkPipelineLayout result;
    VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutInfo, ALLOCATOR, &result))
    return result;
}


VkPipeline createPipeline(VkDevice device, VkPipelineLayout pipelineLayout, VkShaderModule shaderModule,
                          uint32_t particultes_N, float delta_t, uint32_t work_groups_size_x)
{
    //--------------------------------
    // Specialisation Constants
    //--------------------------------
    #pragma pack(push, 1)
    struct Data { uint32_t particultes_N; float delta_t; uint32_t work_groups_size_x;} data;
    #pragma pack(pop)
    VkSpecializationMapEntry mapEntry[] = {
        {0, offsetof(Data, particultes_N), sizeof(uint32_t)}, // particultes_N
        {1, offsetof(Data, delta_t), sizeof(float)},    // delta_t
        {50, offsetof(Data, work_groups_size_x), sizeof(uint32_t)}    // work_groups_size_x
    };
    data.work_groups_size_x = work_groups_size_x;
    data.particultes_N = particultes_N;
    data.delta_t = delta_t;

    VkSpecializationInfo specInfo {};
    specInfo.pMapEntries = mapEntry;
    specInfo.mapEntryCount = 3;
    specInfo.dataSize = sizeof(Data);
    specInfo.pData = &data;

    //----------------------
    // Shader
    //----------------------
    VkPipelineShaderStageCreateInfo shaderStageInfo {};
    shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageInfo.module = shaderModule;
    shaderStageInfo.pSpecializationInfo = &specInfo;
    shaderStageInfo.pName = "main";

    //--------------------------------------------
    // Piepline
    //--------------------------------------------
    VkComputePipelineCreateInfo pipelineCreateInfo {};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.stage = shaderStageInfo;
    pipelineCreateInfo.layout = pipelineLayout;
    pipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineCreateInfo.basePipelineIndex = -1;

    VkPipeline pipeline;
    VK_CHECK_RESULT(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, ALLOCATOR, &pipeline))
    return pipeline;
}




VkDescriptorPool createDescriptorPool(VkDevice device)
{
    const uint32_t descriptorCount = 1;
    std::array<VkDescriptorPoolSize, 1> poolSizes {};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[0].descriptorCount = descriptorCount;

    VkDescriptorPoolCreateInfo poolInfo {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>( poolSizes.size() );
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = descriptorCount;
    poolInfo.flags = 0;

    VkDescriptorPool result;
    VK_CHECK_RESULT(vkCreateDescriptorPool(device, &poolInfo, ALLOCATOR, &result))
    return result;
}


VkDescriptorSet allocateDescriptorSet(VkDevice device,
                                      VkDescriptorPool descriptorPool,
                                      VkDescriptorSetLayout descriptorSetLayout)
{
    std::vector<VkDescriptorSetLayout> layouts {descriptorSetLayout};

    VkDescriptorSetAllocateInfo allocInfo {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(layouts.size());
    allocInfo.pSetLayouts = layouts.data();

    VkDescriptorSet result;
    VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &result))
    return result;
}

// The descriptor sets have been allocated now, but the descriptors within still need to be configured.
// This is the method to update state of descriptors
void updateDescriptorSets(VkDevice device, VkDescriptorSet descriptorSet, VkBuffer buffer) {
    // configure
    VkDescriptorBufferInfo bufferInfo {};
    bufferInfo.buffer = buffer;
    bufferInfo.offset = 0;
    bufferInfo.range = VK_WHOLE_SIZE;

    // update state
    std::array<VkWriteDescriptorSet, 1> descriptorWrites {};

    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = descriptorSet;
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].dstArrayElement = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pBufferInfo = &bufferInfo;
    descriptorWrites[0].pImageInfo = nullptr; // Optional
    descriptorWrites[0].pTexelBufferView = nullptr; // Optional

    vkUpdateDescriptorSets(device,
                           static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(),
                           0, nullptr);
}


void recordCommandBuffer(uint32_t dispatch_groups,
                         VkCommandBuffer cmdBuf,
                         VkPipeline pipeline_acc_vel,
                         VkPipeline pipeline_pos,
                         VkPipelineLayout pipelineLayout,
                         VkDescriptorSet descriptorSet,
                         VkBuffer buffer)
{
    VkCommandBufferBeginInfo beginInfo {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    beginInfo.pInheritanceInfo = nullptr; // Optional
    VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuf, &beginInfo))
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_acc_vel);
    vkCmdDispatch(cmdBuf, dispatch_groups, 1, 1);

    VkBufferMemoryBarrier barrier {};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buffer;
    barrier.offset = 0;
    barrier.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmdBuf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, // dependencyFlags
                         0, nullptr,  // memoryBarrier
                         1, &barrier, // bufferMemoryBarrier
                         0, nullptr   // imageMemoryBarrier
                         );

    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_pos);
    vkCmdDispatch(cmdBuf, dispatch_groups, 1, 1);
    VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuf))

}


void recordCommandBuffer_simple(uint32_t dispatch_groups,
                         VkCommandBuffer cmdBuf,
                         VkPipeline pipeline,
                         VkPipelineLayout pipelineLayout,
                         VkDescriptorSet descriptorSet,
                         VkBuffer buffer)
{
// I assume  that this is not needed in my case as this will be synchronized by Fance
//    VkBufferMemoryBarrier barrier_1 {};
//    barrier_1.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
//    barrier_1.srcAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
//    barrier_1.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
//    barrier_1.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
//    barrier_1.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
//    barrier_1.buffer = buffer;
//    barrier_1.offset = 0;
//    barrier_1.size = VK_WHOLE_SIZE;
//    vkCmdPipelineBarrier(cmdBuf,
//                         VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
//                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
//                         0, // dependencyFlags
//                         0, nullptr,  // memoryBarrier
//                         1, &barrier_1, // bufferMemoryBarrier
//                         0, nullptr   // imageMemoryBarrier
//                         );

    VkBufferMemoryBarrier barrier_2 {};
    barrier_2.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier_2.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier_2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier_2.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier_2.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier_2.buffer = buffer;
    barrier_2.offset = 0;
    barrier_2.size = VK_WHOLE_SIZE;

    VkCommandBufferBeginInfo beginInfo {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    beginInfo.pInheritanceInfo = nullptr; // Optional
    VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuf, &beginInfo))

    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdDispatch(cmdBuf, dispatch_groups, 1, 1);
    vkCmdPipelineBarrier(cmdBuf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
                         0, // dependencyFlags
                         0, nullptr,  // memoryBarrier
                         1, &barrier_2, // bufferMemoryBarrier
                         0, nullptr   // imageMemoryBarrier
                         );
    VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuf))
}

} // namespace compute
