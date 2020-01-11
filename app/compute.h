#pragma once
#include "common.h"

namespace compute {
VkDescriptorSetLayout createDescriptorSetLayout(VkDevice device) ;
VkPipelineLayout createPipelineLayout(VkDevice device, VkDescriptorSetLayout descriptorSetLayout);
VkPipeline createPipeline(VkDevice device, VkPipelineLayout pipelineLayout, VkShaderModule shaderModule,
                          uint32_t particultes_N, float delta_t, uint32_t work_groups_size_x);
VkDescriptorPool createDescriptorPool(VkDevice device);
VkDescriptorSet allocateDescriptorSet(VkDevice device,
                                      VkDescriptorPool descriptorPool,
                                      VkDescriptorSetLayout descriptorSetLayout);
void updateDescriptorSets(VkDevice device, VkDescriptorSet descriptorSet, VkBuffer buffer);

void recordCommandBuffer(uint32_t dispatch_groups,
                         VkCommandBuffer cmdBuf,
                         VkPipeline pipeline_acc_vel,
                         VkPipeline pipeline_pos,
                         VkPipelineLayout pipelineLayout,
                         VkDescriptorSet descriptorSet,
                         VkBuffer buffer);

void recordCommandBuffer_simple(uint32_t dispatch_groups,
                         VkCommandBuffer cmdBuf,
                         VkPipeline pipeline,
                         VkPipelineLayout pipelineLayout,
                         VkDescriptorSet descriptorSet,
                                VkBuffer buffer);
} // namespace compute
