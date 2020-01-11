#pragma once
#include "utils.hpp"
#include <glm/glm.hpp>

namespace graphics {

VkRenderPass createRenderPass(VkDevice device, VkFormat colorAttachmentFormat, VkFormat depthAttachmentFormat);
VkDescriptorSetLayout createDescriptorSetLayout(VkDevice device);
VkDescriptorPool createDescriptorPool(VkDevice device, uint32_t descriptorCount);
std::vector<VkDescriptorSet> allocateDescriptorSets(VkDevice device,
                                                    VkDescriptorPool descriptorPool,
                                                    VkDescriptorSetLayout descriptorSetLayout,
                                                    uint32_t n);
VkPipelineLayout createPipelineLayout(VkDevice device, VkDescriptorSetLayout descriptorSetLayout);
VkPipeline createGraphicsPipeline(VkDevice device,
                                  VkExtent2D extent,
                                  VkRenderPass renderPass,
                                  VkPipelineLayout pipelineLayout,
                                  VkVertexInputBindingDescription pVertexBindingDescriptions,
                                  std::vector<VkVertexInputAttributeDescription> pVertexAttributeDescriptions);

std::vector<VkFramebuffer> createFramebuffers(VkDevice device,
                                              VkRenderPass renderPass,
                                              const std::vector<VkImageView>& imageViews,
                                              VkExtent2D extent,
                                              VkImageView depthImageView);

void recordCommandBufferForFramebuffer(VkCommandBuffer cmdBuf,
                                       VkPipeline pipeline,
                                       VkFramebuffer framebuffer,
                                       VkExtent2D extent,
                                       VkRenderPass renderPass,
                                       VkPipelineLayout pipelineLayout,
                                       VkBuffer vertexBuffer,
                                       uint32_t particulesNumber,
                                       glm::mat4& mvp);
} // namespace: graphics
