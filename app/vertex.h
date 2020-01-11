#pragma once
#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <array>

struct Vertex {
    glm::vec4 pos_m; // vec3: position, vec1: mass
    glm::vec4 velocity; // vec3: velocity, vec1: aligning

    Vertex() {}

    Vertex(glm::vec3 pos, glm::vec3 velocity, glm::f32 weight):
        pos_m{pos, weight}, velocity{velocity, 0.0f}
    {}

    glm::vec3 pos() {
        return glm::vec3{pos_m.x, pos_m.y, pos_m.z};
    }

    void setPos(const glm::vec3& pos) {
        pos_m.x = pos.x;
        pos_m.y = pos.y;
        pos_m.z = pos.z;
    }

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription {};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 1> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 1> attributeDescriptions = {};

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos_m);

//        attributeDescriptions[1].binding = 0;
//        attributeDescriptions[1].location = 1;
//        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
//        attributeDescriptions[1].offset = offsetof(Vertex, velocity);

//        attributeDescriptions[2].binding = 0;
//        attributeDescriptions[2].location = 2;
//        attributeDescriptions[2].format = VK_FORMAT_R32_SFLOAT;
//        attributeDescriptions[2].offset = offsetof(Vertex, weight);

        return attributeDescriptions;
    }
};
