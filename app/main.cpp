#include <iostream>
#include <chrono>
#include <random>
#include <cmath>
#include <functional>
#include "vkcore.hpp"
#include "compute.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/string_cast.hpp>

#include "vertex.h"
#include "graphics.h"
#include "test_alg.h"

constexpr float pi() { return std::atan(1.0f) * 4; }

namespace  {
VkExtent2D fb_size;
}



struct vkcore {
    VkInstance instance;
#ifdef ENABLE_VALIDATION_LAYERS
    VkDebugUtilsMessengerEXT debugMessenger;
#endif
    PhysicalDevice physicalDevice;
    VkSurfaceKHR surface;
    QueueFamilyIndices queueFamilyIndices;
    VkDevice device;
    VkQueue graphicsQueue;
    VkCommandPool graphicsCommandPool;
    SwapChain swapChain;

    vkcore(GLFWwindow* window) {
        instance = createVulkanInstance( getRequiredInstanceExtensions(), getRequiredValidationLayers() );
#ifdef ENABLE_VALIDATION_LAYERS
        debugMessenger = setupDebugMessenger(instance);
#endif
        physicalDevice = pickPhysicalDevice(instance, getRequiredDeviceExtensions());

        surface = createSurface(instance, window);
        queueFamilyIndices = physicalDevice.findQueueFamilyIndices(surface);
        device = physicalDevice.createDevice( getRequiredDeviceExtensions(),
                                              queueFamilyIndices.uniqueIndices() );
        vkGetDeviceQueue(device, queueFamilyIndices.graphicsComputeFamily, 0, &graphicsQueue); // 0 - as we created only one queue
        graphicsCommandPool = createCommandPool(device, queueFamilyIndices.graphicsComputeFamily);
        assert( queueFamilyIndices.presentFamily == queueFamilyIndices.graphicsComputeFamily );

        swapChain = createSwapChain(device, physicalDevice, surface, queueFamilyIndices, fb_size);

    }

    ~vkcore() {
        swapChain.destroy(device);
        vkDestroyCommandPool(device, graphicsCommandPool, ALLOCATOR);
        vkDestroySurfaceKHR(instance, surface, ALLOCATOR);
        vkDestroyDevice(device, ALLOCATOR);
    #ifdef ENABLE_VALIDATION_LAYERS
        destroyDebugMessager(instance, debugMessenger);
    #endif
        vkDestroyInstance(instance, ALLOCATOR);
    }
};


//=============================
// GLFW
//=============================

VkExtent2D getFramebufferSizeGLFW(GLFWwindow *window) {
    int window_width, window_height;
    glfwGetFramebufferSize(window, &window_width, &window_height);
    if (window_width == 0) {
        // it means that we have and error
        throw std::runtime_error("failed to get window size!");
    }

    return VkExtent2D { static_cast<uint32_t>(window_width),
                        static_cast<uint32_t>(window_height) };
}


void initWindow(std::function<void(GLFWwindow*)> callback) {
    if (!glfwInit()) {
        throw "GLFW initialization failed!";
    }
    constexpr VkExtent2D initial_window_size {1980, 1080};
    //constexpr VkExtent2D initial_window_size {2980, 2000};
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(initial_window_size.width, initial_window_size.height,
                                          "Multibody", nullptr, nullptr);
    fb_size = getFramebufferSizeGLFW(window);
//    glfwSetFramebufferSizeCallback(window, framebufferResizeCallbackGLFW);
//    glfwSetWindowIconifyCallback(window, windowMinimizedCallbackGLFW);
//    glfwSetKeyCallback(window, keyCallbackGLFW);
//    glfwSetMouseButtonCallback(window, mouseButtonCallbackGLFW);
//    glfwSetCursorPosCallback(window, cursorPositionCallbackGLFW);
    callback(window);

    glfwDestroyWindow(window);
    glfwTerminate();
}

//=============================
// Vulkan
//=============================


struct SyncObjects {
    VkSemaphore imageAvailableSemaphore;// = createSemaphore(device);
    VkSemaphore renderFinishedSemaphore;// = createSemaphore(device);
    VkFence fence;

    void destroy(VkDevice device) {
        vkDestroySemaphore(device, imageAvailableSemaphore, ALLOCATOR);
        vkDestroySemaphore(device, renderFinishedSemaphore, ALLOCATOR);
        vkDestroyFence(device, fence, ALLOCATOR);
    }
};


Image createDepthImageResource(vkcore& ctx, VkExtent3D extent)
{
    VkFormat format = ctx.physicalDevice.findDepthFormat();
    Image result = createDepthImage(ctx.device, extent, format);
    transitionDepthImageToOptimal(ctx.device, ctx.graphicsCommandPool, ctx.graphicsQueue, result.image, format);
    return result;
}




//==========================
// Drawing
//==========================

void submitCommandBufferToGraphicsQueue(VkQueue graphicsQueue,
                         VkCommandBuffer cmdBuf,
                         SyncObjects& s)
{
    VkSemaphore waitSemaphores[] = {s.imageAvailableSemaphore};
    VkSemaphore signalSemaphores[] = {s.renderFinishedSemaphore};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuf;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    VK_CHECK_RESULT(vkQueueSubmit(graphicsQueue, 1, &submitInfo, s.fence))
}


void vk_display_info() {
    std::cout << "getRequiredInstanceExtensions()\n";
    for (const auto& e: getRequiredInstanceExtensions()) {
        std::cout << "  " << e << '\n';
    }
    std::cout << "getRequiredDeviceExtensions()\n";
    for (const auto& e: getRequiredDeviceExtensions()) {
        std::cout << "  " << e << '\n';
    }
    std::cout << "getRequiredValidationLayers()\n";
    for (const auto& e: getRequiredValidationLayers()) {
        std::cout << "  " << e << '\n';
    }
}






void initParticleBuffer_ring_corners(Vertex* data, std::size_t N) {
    glm::vec3 v {0.0f};

    data[0] = Vertex{glm::vec3{1.0f, 1.0f, -1.0f}, v, 0.0f};
    data[1] = Vertex{glm::vec3{-1.0f, 1.0f, -1.0f}, v, 0.0f};
    data[2] = Vertex{glm::vec3{-1.0f, -1.0f, -1.0f}, v, 0.0f};
    data[3] = Vertex{glm::vec3{1.0f, -1.0f, -1.0f}, v, 0.0f};

    data[4] = Vertex{glm::vec3{1.0f, 1.0f, 1.0f}, v, 0.0f};
    data[5] = Vertex{glm::vec3{-1.0f, 1.0f, 1.0f}, v, 0.0f};
    data[6] = Vertex{glm::vec3{-1.0f, -1.0f, 1.0f}, v, 0.0f};
    data[7] = Vertex{glm::vec3{1.0f, -1.0f, 1.0f}, v, 0.0f};

//    const float a = 0.5;
//    data[0] = Vertex{glm::vec3{a, a, 0.0f}, v, 0.0f};
//    data[1] = Vertex{glm::vec3{-a, a, 0.0f}, v, 0.0f};
//    data[2] = Vertex{glm::vec3{-a, -a, 0.0f}, v, 0.0f};
//    data[3] = Vertex{glm::vec3{a, -a, 0.0f}, v, 0.0f};

//    data[4] = Vertex{glm::vec3{a, a, 1.0f}, v, 0.0f};
//    data[5] = Vertex{glm::vec3{-a, a, 1.0f}, v, 0.0f};
//    data[6] = Vertex{glm::vec3{-a, -a, 1.0f}, v, 0.0f};
//    data[7] = Vertex{glm::vec3{a, -a, a}, v, 0.0f};
    // This defines ring

    std::size_t Ns = N - 8;
    for (size_t i = 0; i < Ns; i++)
    {
        // float(i)/(N+1)
        float alpha = 2.0f * pi() * i/Ns;
        float z = 2.0f*float(i)/Ns - 1.0f; // -1..1
        data[i + 8] = Vertex{
                         glm::vec3{std::sin(alpha), std::cos(alpha), z},
                         18.0f * glm::vec3{std::cos(alpha), -std::sin(alpha), 0.0f},
                         0.0f
                      };
    }
}


void initParticleBuffer_ring(Vertex* data, std::size_t N) {
    glm::mat4 r = glm::mat4(1.0f);
    //ubo.model = glm::translate(glm::mat4(1.0f), glm::vec3 {0.0f, 0.0f, 0.0f});
    r = glm::rotate(pi() * -0.25f, glm::vec3(1.0f, 0.0f, 0.0f));
    for (std::size_t i = 0; i < N; i++)
    {
        float alpha = 2.0f * pi() * i/N;
        glm::vec4 pos = 0.7f * glm::vec4{std::sin(alpha), std::cos(alpha), 0.0f, 0.0f};
        glm::vec4 vel = 1.0f * glm::vec4{std::cos(alpha), -std::sin(alpha), 0.0f, 0.0f};
        pos = r * pos;
        vel = r * vel;
        data[i] = Vertex{glm::vec3{pos.x, pos.y, pos.z},
                         glm::vec3{vel.x, vel.y, vel.z},
                         0.03f};
    }
}


void initParticleBuffer_ring_flat(Vertex* data, std::size_t N) {
    for (std::size_t i = 0; i < N; i++)
    {
        float alpha = 2.0f * pi() * i/N;
        glm::vec4 pos = 10.0f * glm::vec4{std::sin(alpha), std::cos(alpha), 0.0f, 0.0f};
        // 2.466f - Hamiltonian < 0 but only little
        glm::vec4 vel = 1.5f * glm::vec4{std::cos(alpha), -std::sin(alpha), 0.0f, 0.0f};
        data[i] = Vertex{glm::vec3{pos.x, pos.y, pos.z},
                         glm::vec3{vel.x, vel.y, vel.z},
                         0.3f};
    }
}



void initParticleBuffer_2_body(Vertex* data, std::size_t N) {
    assert(N == 2);
    data[0] = Vertex{glm::vec3{0.0f, 0.0f, 0.0f},
                     glm::vec3{0.0f, 0.0f, 0.0f},
                     100.0f};
    data[1] = Vertex{glm::vec3{10.0f, 0.0f, 0.0f},
                     glm::vec3{0.0f, 0.0f, 0.0f},
                     1.0f};
}



void initParticleBuffer_galaxy_no_rotation_flat(Vertex* data, std::size_t N) {
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator{seed};
    std::uniform_real_distribution<float> distribution_angle(0.0f, 2.0f * pi());
    const float sigma = 1.0f;
    std::normal_distribution<float> distribution_distance(0.0f, sigma);

    auto genDistance = [&]() -> float {
        while (true) {
            float res = distribution_distance(generator);
            if (-1.0f < res && res < 1.0f) {
                return res;
            }
        }
    };

    auto genAngle = [&]() -> float {
        return distribution_angle(generator);
    };

    glm::mat4 r = glm::mat4(1.0f);
    //ubo.model = glm::translate(glm::mat4(1.0f), glm::vec3 {0.0f, 0.0f, 0.0f});
    r = glm::rotate(pi(), glm::vec3(1.0f, 0.0f, 0.0f));

    for (std::size_t i = 0; i < N; i++)
    {
        float alpha = genAngle();
        float r = genDistance();
        glm::vec4 pos = r * glm::vec4{std::sin(alpha), std::cos(alpha), 0.0f, 0.0f};
        glm::vec4 vel = glm::vec4{0.0f};
        pos = r * pos;
        vel = r * vel;
        data[i] = Vertex{glm::vec3{pos.x, pos.y, pos.z},
                         glm::vec3{vel.x, vel.y, vel.z},
                         0.03f};
    }
}


void initParticleBuffer_galaxy_2d(Vertex* data, std::size_t N) {
    //auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator{0};
    std::uniform_real_distribution<float> distribution_angle(0.0f, 2.0f * pi());
    const float sigma = 0.2f;
    std::normal_distribution<float> distribution_distance(0.0f, 1.0f);

    auto genDistance = [&]() -> float {
        return std::abs(distribution_distance(generator));
//        while (true) {
//            float res = std::abs(distribution_distance(generator));
//            if (res < 1.0f) {
//                return res;
//            }
//        }
    };

    auto genAngle = [&]() -> float {
        return distribution_angle(generator);
    };

    auto cdfNorm = [sigma](float x) -> float {
        auto s = sigma * std::sqrt(2.0f);
        return 0.5f*(1.0f + std::erf(x/s));
    };

    auto cIns = [cdfNorm](float x) -> float {
        return cdfNorm(std::abs(x)) - cdfNorm(-std::abs(x));
    };

    const float mass = 0.01f;

    glm::mat4 r = glm::mat4(1.0f);
    //ubo.model = glm::translate(glm::mat4(1.0f), glm::vec3 {0.0f, 0.0f, 0.0f});
    r = glm::rotate(0.3f * pi(), glm::vec3(1.0f, 0.0f, 0.0f));

    for (std::size_t i = 0; i < N; i++)
    {
        float alpha = genAngle();
        float dist = genDistance();
        glm::vec4 pos = dist * glm::vec4{std::sin(alpha), std::cos(alpha), 0.0f, 0.0f};
        glm::vec4 vel = glm::vec4{std::cos(alpha), -std::sin(alpha), 0.0f, 0.0f};
        pos = r * pos;
        vel = 0.006f * float(N) * mass * cIns(dist) * r * vel;
        data[i] = Vertex{glm::vec3{pos.x, pos.y, pos.z},
                         glm::vec3{vel.x, vel.y, vel.z},
                         mass};
    }
}


void initParticleBuffer_galaxy_3d(Vertex* data, std::size_t N) {
    //auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator{0};
    std::uniform_real_distribution<float> distribution_angle(0.0f, 2.0f * pi());
    const float sigma = 0.2f;
    std::normal_distribution<float> distribution_distance(0.0f, 1.0f);

    auto genDistance = [&]() -> float {
        return std::abs(distribution_distance(generator));
//        while (true) {
//            float res = std::abs(distribution_distance(generator));
//            if (res < 1.0f) {
//                return res;
//            }
//        }
    };

    auto genAngle = [&]() -> float {
        return distribution_angle(generator);
    };

    auto cdfNorm = [sigma](float x) -> float {
        auto s = sigma * std::sqrt(2.0f);
        return 0.5f*(1.0f + std::erf(x/s));
    };

    auto cIns = [cdfNorm](float x) -> float {
        return cdfNorm(std::abs(x)) - cdfNorm(-std::abs(x));
    };

    const float mass = 0.05f;

    glm::mat4 r = glm::mat4(1.0f);
    //ubo.model = glm::translate(glm::mat4(1.0f), glm::vec3 {0.0f, 0.0f, 0.0f});
    r = glm::rotate(0.4f * pi(), glm::vec3(1.0f, 0.0f, 0.0f));

    for (std::size_t i = 0; i < N; i++)
    {
        float alpha = genAngle();
        float dist = genDistance();
        glm::vec4 pos = dist * glm::vec4{std::sin(alpha), std::cos(alpha), 0.0f, 0.0f};
        glm::vec4 vel = glm::vec4{std::cos(alpha), -std::sin(alpha), 0.0f, 0.0f};
        pos = r * pos;
        vel = 0.002f * float(N) * mass * cIns(dist) * r * vel;
        data[i] = Vertex{glm::vec3{pos.x, pos.y, pos.z},
                         glm::vec3{vel.x, vel.y, vel.z},
                         mass};
    }

//    data[0] = Vertex{glm::vec3{2.0f, 3.0f, 0.0f},
//                     glm::vec3{0.0f, -0.2f, 0.0f},
//                     300};

}

class ParticleBuffer_cpu {
public:
    ParticleBuffer_cpu(VkDevice device, std::size_t N = 5): device{device}, N{N}
    {
        VkDeviceSize msize = sizeof(Vertex) * N;
        vertexBuffer = Buffer {device, msize,
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    PhysicalDevice::amd_memory_type::host_local_cached_3};

        vkMapMemory(device, vertexBuffer.memory, 0, msize, 0, reinterpret_cast<void**>(&data));
        initParticleBuffer_ring_corners(data, N);
    }

    ~ParticleBuffer_cpu() {
        vkUnmapMemory(device, vertexBuffer.memory);
    }

    void update() {
        auto bodyInteraction = [](glm::vec3 bi, glm::vec3 bj, float mj, glm::vec3 ai) -> glm::vec3 {
            glm::vec3 r = bj - bi;
            const float EPS2 = 0.05f;
            float distSqr = glm::dot(r, r) + EPS2;
            float distSixth = distSqr * distSqr * distSqr;
            float invDistCube = 1.0f / sqrtf(distSixth);
            float s = mj * invDistCube;
            ai += r * s;
            return ai;
        };

        const float t_delta = 1.0f / 1000;

        // calculate acc for each body
//        for (size_t i = 0; i < N; i++)
//        {
//            glm::vec3 a(0);
//            for (size_t j = 0; j < N; j++)
//            {
//                a = bodyInteraction(data[i].pos_m, data[j].pos_m, data[j].pos_m.w, a);
//            }
//            data[i].velocity += t_delta * a;
//            data[i].pos_m += t_delta * data[i].velocity;
//        }
    }
private:
    VkDevice device;
    Vertex* data;
public:
    Buffer vertexBuffer;
    std::size_t N;
};




Buffer createAndInitGPUParticleBuffer(vkcore& ctx, std::size_t N) {
    VkDeviceSize msize = sizeof(Vertex) * N;
    Buffer buffer_stage {ctx.device, msize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                PhysicalDevice::amd_memory_type::host_local_primary_1};

    // Init stage buffer
    Vertex* data;
    vkMapMemory(ctx.device, buffer_stage.memory, 0, msize, 0, reinterpret_cast<void**>(&data));
    initParticleBuffer_galaxy_3d(data, N);
    //initParticleBuffer_ring_flat(data, N);
    vkUnmapMemory(ctx.device, buffer_stage.memory);

    // Copy buffer to the final
    Buffer buffer {ctx.device, msize,
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                PhysicalDevice::amd_memory_type::device_local_primary_0};

    copyBuffer(ctx.device, ctx.graphicsQueue, ctx.graphicsCommandPool, buffer_stage.buffer, buffer.buffer, msize);
    return buffer;
}


class ParticleBuffer_gpu {
public:
    ParticleBuffer_gpu(vkcore& ctx, std::size_t N, std::size_t work_groups_size_x):
        device{ctx.device},
        vertexBuffer {createAndInitGPUParticleBuffer(ctx, N)},
        N{N}
    {
        float delta_t = 0.001f;

        VkDescriptorSetLayout descriptorSetLayout = compute::createDescriptorSetLayout(device);
        VkPipelineLayout pipelineLayout = compute::createPipelineLayout(device, descriptorSetLayout);

        VkDescriptorPool descriptorPool = compute::createDescriptorPool(device);
        VkDescriptorSet descriptorSet = compute::allocateDescriptorSet(device, descriptorPool, descriptorSetLayout);

        // Pipeline to calculate Acceleration and then update Velocities
        VkShaderModule shaderModule_acc_vel = createShaderModule(device, SHADERS_DIR"/compute_acc.comp.spv");
        VkPipeline pipeline_acc_vel = compute::createPipeline(device, pipelineLayout, shaderModule_acc_vel, static_cast<uint32_t>(N), delta_t, work_groups_size_x);
        vkDestroyShaderModule(device, shaderModule_acc_vel, ALLOCATOR);

        // Pipeline to update Position (now we have new Velocities)
        VkShaderModule shaderModule_pos = createShaderModule(device, SHADERS_DIR"/compute_pos.comp.spv");
        VkPipeline pipeline_pos = compute::createPipeline(device, pipelineLayout, shaderModule_pos, static_cast<uint32_t>(N), delta_t, work_groups_size_x);
        vkDestroyShaderModule(device, shaderModule_pos, ALLOCATOR);

        compute::updateDescriptorSets(device, descriptorSet, vertexBuffer.buffer);

        this->descriptorSetLayout = descriptorSetLayout;
        this->pipelineLayout = pipelineLayout;
        this->descriptorPool = descriptorPool;
        this->descriptorSet = descriptorSet;
        this->pipeline_acc_vel = pipeline_acc_vel;
        this->pipeline_pos = pipeline_pos;
    }

    void recordCommandBuffer(VkCommandBuffer cmdBuf, uint32_t dispatch_groups) {
        // VkCommandBuffer cmdBuf = allocateCommandBuffer(device, ctx.graphicsCommandPool);
        compute::recordCommandBuffer(dispatch_groups, cmdBuf, pipeline_acc_vel, pipeline_pos, pipelineLayout, descriptorSet, vertexBuffer.buffer);
    }

    ~ParticleBuffer_gpu() {
        // free cmd buffer?
        // free descriptor set?
        vkDestroyDescriptorPool(device, descriptorPool, ALLOCATOR);
        vkDestroyPipeline(device, pipeline_acc_vel, ALLOCATOR);
        vkDestroyPipeline(device, pipeline_pos, ALLOCATOR);
        vkDestroyPipelineLayout(device, pipelineLayout, ALLOCATOR);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, ALLOCATOR);
    }

private: // Local
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;
    VkPipeline pipeline_acc_vel;
    VkPipeline pipeline_pos;
private:  // Global
    VkDevice device;
public:
    Buffer vertexBuffer;
    std::size_t N;
};


glm::mat4 calcMVP() {
    glm::mat4 model = glm::mat4(1.0f);
    //ubo.model = glm::translate(glm::mat4(1.0f), glm::vec3 {0.0f, 0.0f, 0.0f});
    //ubo.model = glm::rotate(ubo.model, 0.5f * time, glm::vec3(0.0f, 1.0f, 0.0f));
    //ubo.model = glm::rotate(ubo.model, glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    model = glm::scale(model, glm::vec3 {1.0f/15});
    //ubo.model[1][1] *= -1;
    // Something is wrong with this function...

    glm::mat4 view = glm::lookAt(glm::vec3{0.0f, 0.0f, -2.0f},
                           glm::vec3{0.0f, 0.0f, 0.0f},
                           glm::vec3(0.0f, -1.0f, 0.0f));



    //ubo.view = glm::rotate(ubo.view, glm::radians(180.0f), glm::vec3(1.0f, 1.0f, 0.0f));
//    ubo.view[1][1] *= -1.0f;

    // To działa ok!
//    ubo.view = glm::rotate(glm::mat4(1.0f), camera.angle_y, glm::vec3 {0.0f, 1.0f, 0.0f});
//    ubo.view = glm::translate(ubo.view, -camera.pos); // lookAt implementation

    //ubo.view =  glm::mat4(1.0f);
    float aspectRatio = static_cast<float>(fb_size.width) / fb_size.height;
    glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspectRatio, 0.2f, 10.0f);
    return proj * view * model;
}


void runGraphics(GLFWwindow* window)
{
    vkcore ctx{window};
    std::size_t N = 1024*40;
    std::size_t work_group_size = 64;
    std::size_t dispatch_groups = N / work_group_size;
    assert(dispatch_groups * work_group_size == N);
    ParticleBuffer_gpu vertexBuffer{ctx, N, work_group_size};
    auto depthFormat = ctx.physicalDevice.findDepthFormat();
    VkDescriptorSetLayout descriptorSetLayout = graphics::createDescriptorSetLayout(ctx.device);
    VkPipelineLayout pipelineLayout = graphics::createPipelineLayout(ctx.device, descriptorSetLayout);

    VkRenderPass renderPass = graphics::createRenderPass(ctx.device, ctx.swapChain.imageFormat, depthFormat);

    auto vertexInputAttributeDescriptionArr = Vertex::getAttributeDescriptions();
    std::vector<VkVertexInputAttributeDescription> vertexInputAttributeDescription;
    vertexInputAttributeDescription.assign(vertexInputAttributeDescriptionArr.cbegin(),
                                           vertexInputAttributeDescriptionArr.cend());
    VkPipeline pipeline = graphics::createGraphicsPipeline(ctx.device, fb_size, renderPass, pipelineLayout,
                                                           Vertex::getBindingDescription(), vertexInputAttributeDescription);

    VkExtent3D fb_size_3d {ctx.swapChain.extent.width, ctx.swapChain.extent.height, 1}; // TODO: why not fb_size??
    Image depthImage = createDepthImageResource(ctx, fb_size_3d);
    std::vector<VkFramebuffer> framebuffers = graphics::createFramebuffers(ctx.device,
                                                                 renderPass,
                                                                 ctx.swapChain.imageViews,
                                                                 ctx.swapChain.extent,
                                                                 depthImage.imageView);
    auto framebufferCount = static_cast<uint32_t>(framebuffers.size());
    std::vector<VkCommandBuffer> buffers = allocateCommandBuffers(ctx.device, ctx.graphicsCommandPool, framebufferCount);
//        VkDescriptorPool descriptorPool = createDescriptorPool(ctx.device, framebufferCount);
//        std::vector<VkDescriptorSet> descriptorSets = allocateDescriptorSets(ctx.device, descriptorPool, descriptorSetLayout, framebufferCount);

    glm::mat4 mvp = calcMVP();

    for (std::size_t i = 0; i < framebufferCount; i++) {
        VkCommandBuffer cmdBuf = buffers[i];
        VkFramebuffer framebuffer = framebuffers[i];
        graphics::recordCommandBufferForFramebuffer(cmdBuf, pipeline, framebuffer, ctx.swapChain.extent,
                                          renderPass, pipelineLayout, vertexBuffer.vertexBuffer.buffer,
                                          vertexBuffer.N, mvp);
    }

    const int MAX_FRAMES_IN_FLIGHT = 1; // With 2 performance is a bit better
    std::vector<SyncObjects> syncObjects(MAX_FRAMES_IN_FLIGHT);
    for (auto& s: syncObjects) {
        s.imageAvailableSemaphore = createSemaphore(ctx.device);
        s.renderFinishedSemaphore = createSemaphore(ctx.device);
        s.fence = createFenceSignaled(ctx.device);
    }

    VkCommandBuffer cmdComputeBuf = allocateCommandBuffer(ctx.device, ctx.graphicsCommandPool);
    vertexBuffer.recordCommandBuffer(cmdComputeBuf, dispatch_groups);

    size_t n = 0;
    size_t currentFrame = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto prev_time = start_time;
    auto curr_time = start_time;
    int report_sec = 1;
    while(!glfwWindowShouldClose(window)) {
        glfwPollEvents();

//            while (window_minimized) {
//                glfwWaitEvents();
//            }

//            if ( new_fb_size.has_value() ) {
//                // Framebuffer size has changed: recreate swapchain and related stuff
//                // TODO: is it the correct place?
//                vkDeviceWaitIdle(ctx.device); // TODO: do we need it?

//                fb_size = new_fb_size.value();
//                new_fb_size.reset();

//                swsDestroy();
//                swsCreate();
//            }

        auto& syncObject = syncObjects[currentFrame];
        vkWaitForFences(ctx.device, 1, &syncObject.fence, VK_TRUE, std::numeric_limits<uint64_t>::max());
        vkResetFences(ctx.device, 1, &syncObject.fence);

        uint32_t imageIndex = acquireNextImageFromSwapchain(ctx.device, ctx.swapChain.swapChain, syncObject.imageAvailableSemaphore);

        //vertexBuffer.update();
        submitCommandBuffer(ctx.graphicsQueue, cmdComputeBuf);

        VkCommandBuffer buffer = buffers[imageIndex];
        submitCommandBufferToGraphicsQueue(ctx.graphicsQueue, buffer, syncObject);

        presentFrame(ctx.graphicsQueue, ctx.swapChain.swapChain, imageIndex, syncObject.renderFinishedSemaphore);
        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
        n++;

        auto stop_time = std::chrono::high_resolution_clock::now();
        auto duration = stop_time - start_time;

        if (int(std::floor(duration.count() * 1e-9)) == report_sec) {
            std::cout << report_sec << '\n';
            report_sec++;
        }

        //curr_time = std::chrono::high_resolution_clock::now();
        //float time_delta = std::chrono::duration<float, std::chrono::seconds::period>(curr_time - prev_time).count();
        //prev_time = curr_time;
        //float curr_time_absolute = std::chrono::duration<float, std::chrono::seconds::period>(curr_time - start_time).count();
    }
    auto stop_time = std::chrono::high_resolution_clock::now();
    vkDeviceWaitIdle(ctx.device);
    auto duration = stop_time - start_time;
    double avgTime = duration.count() / n;  // nanoseconds
    std::cout << "avg time: " << 1e-3 * avgTime  << "us/frame  FPS: " << 1e9/avgTime << "  Frames: " << n << "\n";

    vkQueueWaitIdle(ctx.graphicsQueue);

//        vkDestroyDescriptorPool(ctx.device, descriptorPool, ALLOCATOR);
    for (auto& so: syncObjects) {
        so.destroy(ctx.device);
    }
    vkFreeCommandBuffers(ctx.device, ctx.graphicsCommandPool, static_cast<uint32_t>(buffers.size()), buffers.data());

    for (auto& fb: framebuffers) {
        vkDestroyFramebuffer(ctx.device, fb, ALLOCATOR);
    }
    depthImage.destroy(ctx.device);
    vkDestroyPipeline(ctx.device, pipeline, ALLOCATOR);
    vkDestroyPipelineLayout(ctx.device, pipelineLayout, ALLOCATOR);
    vkDestroyDescriptorSetLayout(ctx.device, descriptorSetLayout, ALLOCATOR);
    vkDestroyRenderPass(ctx.device, renderPass, ALLOCATOR);
}





void testCompute(GLFWwindow* window) {
    vkcore ctx{window};

    // TILE_SIZE = 32; // workgroup size
    std::size_t N = 128;
    uint32_t work_group_size = 1024;
    uint32_t dispatch_groups = 2;
    float delta_t = 2;

    auto createBuffer = [&ctx](std::size_t N) -> Buffer {
        VkDeviceSize msize = sizeof(Vertex) * N;
        Buffer buffer {ctx.device, msize,
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    PhysicalDevice::amd_memory_type::host_local_primary_1};

        Vertex* data;
        vkMapMemory(ctx.device, buffer.memory, 0, msize, 0, reinterpret_cast<void**>(&data));

        for (std::size_t i = 0; i < N; i++) {
            auto& v = data[i];
            //v.pos_m = glm::vec4(float(i));
            v.pos_m = glm::vec4(1.0);
            v.velocity = glm::vec4(0.0f);
        }

        vkUnmapMemory(ctx.device, buffer.memory);
        return buffer;
    };

    auto displayBuffer = [&ctx](Buffer& buffer, std::size_t N) {
        VkDeviceSize msize = sizeof(Vertex) * N;
        Vertex* data;
        vkMapMemory(ctx.device, buffer.memory, 0, msize, 0, reinterpret_cast<void**>(&data));

        for (std::size_t i = 0; i < N; i++) {
            auto& v = data[i];
            std::cout << i << ": pos(" << v.pos_m.x << ", " << v.pos_m.y << ", " << v.pos_m.z << ")";
            std::cout <<      "  vel(" << v.velocity.x << ", " << v.velocity.y << ", " << v.velocity.z << ")  ";
            std::cout << "  mass: " << v.pos_m.w << '\n';
        }

        vkUnmapMemory(ctx.device, buffer.memory);
    };

    VkDescriptorSetLayout descriptorSetLayout = compute::createDescriptorSetLayout(ctx.device);
    VkPipelineLayout pipelineLayout = compute::createPipelineLayout(ctx.device, descriptorSetLayout);

    VkDescriptorPool descriptorPool = compute::createDescriptorPool(ctx.device);
    VkDescriptorSet descriptorSet = compute::allocateDescriptorSet(ctx.device, descriptorPool, descriptorSetLayout);

    // Pipeline to calculate Acceleration and then update Velocities
    VkShaderModule shaderModule_acc_vel = createShaderModule(ctx.device, SHADERS_DIR"/compute_acc.comp.spv");
    VkPipeline pipeline_acc_vel = compute::createPipeline(ctx.device, pipelineLayout, shaderModule_acc_vel, static_cast<uint32_t>(N), delta_t, work_group_size);
    vkDestroyShaderModule(ctx.device, shaderModule_acc_vel, ALLOCATOR);

    // Pipeline to update Position (now we have new Velocities)
    VkShaderModule shaderModule_pos = createShaderModule(ctx.device, SHADERS_DIR"/compute_pos.comp.spv");
    VkPipeline pipeline_pos = compute::createPipeline(ctx.device, pipelineLayout, shaderModule_pos, static_cast<uint32_t>(N), delta_t, work_group_size);
    vkDestroyShaderModule(ctx.device, shaderModule_pos, ALLOCATOR);


    Buffer buffer = createBuffer(N);
    compute::updateDescriptorSets(ctx.device, descriptorSet, buffer.buffer);

    VkCommandBuffer cmdBuf = allocateCommandBuffer(ctx.device, ctx.graphicsCommandPool);
    compute::recordCommandBuffer(dispatch_groups, cmdBuf, pipeline_acc_vel, pipeline_pos, pipelineLayout, descriptorSet, buffer.buffer);
    VkFence fence = createFence(ctx.device);

    submitCommandBuffer(ctx.graphicsQueue, cmdBuf, fence);
    vkWaitForFences(ctx.device, 1, &fence, VK_TRUE, UINT64_MAX);
    displayBuffer(buffer, N);


    vkDestroyFence(ctx.device, fence, ALLOCATOR);
    //vkFreeCommandBuffers(ctx.device, ctx.graphicsCommandPool, 1, &cmd);
    vkDestroyDescriptorPool(ctx.device, descriptorPool, ALLOCATOR);
    vkDestroyPipeline(ctx.device, pipeline_acc_vel, ALLOCATOR);
    vkDestroyPipeline(ctx.device, pipeline_pos, ALLOCATOR);
    vkDestroyPipelineLayout(ctx.device, pipelineLayout, ALLOCATOR);
    vkDestroyDescriptorSetLayout(ctx.device, descriptorSetLayout, ALLOCATOR);
}




void testCompute_simple(GLFWwindow* window) {
    vkcore ctx{window};

    // TILE_SIZE = 32; // workgroup size
    std::size_t N = 128;
    std::size_t work_group_size = 1024;
    uint32_t dispatch_groups = 2;
    float delta_t = 2.0f * pi() / 360; // 1 degree

    auto createBuffer = [&ctx](std::size_t N) -> Buffer {
        VkDeviceSize msize = sizeof(Vertex) * N;
        Buffer buffer {ctx.device, msize,
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    PhysicalDevice::amd_memory_type::host_local_primary_1};

        Vertex* data;
        vkMapMemory(ctx.device, buffer.memory, 0, msize, 0, reinterpret_cast<void**>(&data));

        for (std::size_t i = 0; i < N; i++) {
            auto& v = data[i];
            //v.pos_m = glm::vec4(float(i));
            v.pos_m = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
            v.velocity = glm::vec4(0.0f);
        }

        vkUnmapMemory(ctx.device, buffer.memory);
        return buffer;
    };

    auto displayBuffer = [&ctx](Buffer& buffer, std::size_t N) {
        VkDeviceSize msize = sizeof(Vertex) * N;
        Vertex* data;
        vkMapMemory(ctx.device, buffer.memory, 0, msize, 0, reinterpret_cast<void**>(&data));

        for (std::size_t i = 0; i < N; i++) {
            auto& v = data[i];
            std::cout << i << ": pos(" << v.pos_m.x << ", " << v.pos_m.y << ", " << v.pos_m.z << ")";
            std::cout <<      "  vel(" << v.velocity.x << ", " << v.velocity.y << ", " << v.velocity.z << ")  ";
            std::cout << "  mass: " << v.pos_m.w << '\n';
        }

        vkUnmapMemory(ctx.device, buffer.memory);
    };

    VkDescriptorSetLayout descriptorSetLayout = compute::createDescriptorSetLayout(ctx.device);
    VkPipelineLayout pipelineLayout = compute::createPipelineLayout(ctx.device, descriptorSetLayout);

    VkDescriptorPool descriptorPool = compute::createDescriptorPool(ctx.device);
    VkDescriptorSet descriptorSet = compute::allocateDescriptorSet(ctx.device, descriptorPool, descriptorSetLayout);


    VkShaderModule shaderModule = createShaderModule(ctx.device, SHADERS_DIR"/compute_simple.comp.spv");
    VkPipeline pipeline = compute::createPipeline(ctx.device, pipelineLayout, shaderModule, static_cast<uint32_t>(N), delta_t, work_group_size);
    vkDestroyShaderModule(ctx.device, shaderModule, ALLOCATOR);

    Buffer buffer = createBuffer(N);
    compute::updateDescriptorSets(ctx.device, descriptorSet, buffer.buffer);

    VkCommandBuffer cmdBuf = allocateCommandBuffer(ctx.device, ctx.graphicsCommandPool);
    assert(false);
    // This is broken as barrieres were added
    // XXXXXXX compute::recordCommandBuffer_simple(dispatch_groups, cmdBuf, pipeline, pipelineLayout, descriptorSet);
    VkFence fence = createFence(ctx.device);

    submitCommandBuffer(ctx.graphicsQueue, cmdBuf, fence);
    vkWaitForFences(ctx.device, 1, &fence, VK_TRUE, UINT64_MAX);
    displayBuffer(buffer, N);


    vkDestroyFence(ctx.device, fence, ALLOCATOR);
    //vkFreeCommandBuffers(ctx.device, ctx.graphicsCommandPool, 1, &cmd);
    vkDestroyDescriptorPool(ctx.device, descriptorPool, ALLOCATOR);
    vkDestroyPipeline(ctx.device, pipeline, ALLOCATOR);
    vkDestroyPipelineLayout(ctx.device, pipelineLayout, ALLOCATOR);
    vkDestroyDescriptorSetLayout(ctx.device, descriptorSetLayout, ALLOCATOR);
}




void runGraphics_test_star_shader(GLFWwindow* window)
{
    vkcore ctx{window};
    ParticleBuffer_cpu vertexBuffer {ctx.device, 8 + 60};

    auto depthFormat = ctx.physicalDevice.findDepthFormat();
    VkDescriptorSetLayout descriptorSetLayout = graphics::createDescriptorSetLayout(ctx.device);
    VkPipelineLayout pipelineLayout = graphics::createPipelineLayout(ctx.device, descriptorSetLayout);

    VkRenderPass renderPass = graphics::createRenderPass(ctx.device, ctx.swapChain.imageFormat, depthFormat);

    auto vertexInputAttributeDescriptionArr = Vertex::getAttributeDescriptions();
    std::vector<VkVertexInputAttributeDescription> vertexInputAttributeDescription;
    vertexInputAttributeDescription.assign(vertexInputAttributeDescriptionArr.cbegin(),
                                           vertexInputAttributeDescriptionArr.cend());
    VkPipeline pipeline = graphics::createGraphicsPipeline(ctx.device, fb_size, renderPass, pipelineLayout,
                                                           Vertex::getBindingDescription(), vertexInputAttributeDescription);

    VkExtent3D fb_size_3d {ctx.swapChain.extent.width, ctx.swapChain.extent.height, 1}; // TODO: why not fb_size??
    Image depthImage = createDepthImageResource(ctx, fb_size_3d);
    std::vector<VkFramebuffer> framebuffers = graphics::createFramebuffers(ctx.device,
                                                                 renderPass,
                                                                 ctx.swapChain.imageViews,
                                                                 ctx.swapChain.extent,
                                                                 depthImage.imageView);
    auto framebufferCount = static_cast<uint32_t>(framebuffers.size());


    std::vector<VkCommandBuffer> buffers = allocateCommandBuffers(ctx.device, ctx.graphicsCommandPool, framebufferCount);
//        VkDescriptorPool descriptorPool = createDescriptorPool(ctx.device, framebufferCount);
//        std::vector<VkDescriptorSet> descriptorSets = allocateDescriptorSets(ctx.device, descriptorPool, descriptorSetLayout, framebufferCount);

    glm::mat4 mvp = calcMVP();

    for (std::size_t i = 0; i < framebufferCount; i++) {
        VkCommandBuffer cmdBuf = buffers[i];
        VkFramebuffer framebuffer = framebuffers[i];
        graphics::recordCommandBufferForFramebuffer(cmdBuf, pipeline, framebuffer, ctx.swapChain.extent,
                                          renderPass, pipelineLayout, vertexBuffer.vertexBuffer.buffer, vertexBuffer.N, mvp);
    }

    const int MAX_FRAMES_IN_FLIGHT = 1; // With 2 performance is a bit better
    std::vector<SyncObjects> syncObjects(MAX_FRAMES_IN_FLIGHT);
    for (auto& s: syncObjects) {
        s.imageAvailableSemaphore = createSemaphore(ctx.device);
        s.renderFinishedSemaphore = createSemaphore(ctx.device);
        s.fence = createFenceSignaled(ctx.device);
    }

    size_t n = 0;
    size_t currentFrame = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto prev_time = start_time;
    auto curr_time = start_time;
    while(!glfwWindowShouldClose(window)) {
        glfwPollEvents();

//            while (window_minimized) {
//                glfwWaitEvents();
//            }

//            if ( new_fb_size.has_value() ) {
//                // Framebuffer size has changed: recreate swapchain and related stuff
//                // TODO: is it the correct place?
//                vkDeviceWaitIdle(ctx.device); // TODO: do we need it?

//                fb_size = new_fb_size.value();
//                new_fb_size.reset();

//                swsDestroy();
//                swsCreate();
//            }

        auto& syncObject = syncObjects[currentFrame];
        vkWaitForFences(ctx.device, 1, &syncObject.fence, VK_TRUE, std::numeric_limits<uint64_t>::max());
        vkResetFences(ctx.device, 1, &syncObject.fence);

        uint32_t imageIndex = acquireNextImageFromSwapchain(ctx.device, ctx.swapChain.swapChain, syncObject.imageAvailableSemaphore);

        //vertexBuffer.update();

        VkCommandBuffer buffer = buffers[imageIndex];
        submitCommandBufferToGraphicsQueue(ctx.graphicsQueue, buffer, syncObject);

        presentFrame(ctx.graphicsQueue, ctx.swapChain.swapChain, imageIndex, syncObject.renderFinishedSemaphore);
        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
        n++;

        //curr_time = std::chrono::high_resolution_clock::now();
        //float time_delta = std::chrono::duration<float, std::chrono::seconds::period>(curr_time - prev_time).count();
        //prev_time = curr_time;
        //float curr_time_absolute = std::chrono::duration<float, std::chrono::seconds::period>(curr_time - start_time).count();
    }
    auto stop_time = std::chrono::high_resolution_clock::now();
    vkDeviceWaitIdle(ctx.device);
    auto duration = stop_time - start_time;
    double avgTime = static_cast<double>(duration.count()) / n;  // nanoseconds
    std::cout << "avg time: " << 1e-3 * avgTime  << "us/frame  FPS: " << 1e9/avgTime << "  Frames: " << n << "\n";

    vkQueueWaitIdle(ctx.graphicsQueue);

//        vkDestroyDescriptorPool(ctx.device, descriptorPool, ALLOCATOR);
    for (auto& so: syncObjects) {
        so.destroy(ctx.device);
    }
    vkFreeCommandBuffers(ctx.device, ctx.graphicsCommandPool, static_cast<uint32_t>(buffers.size()), buffers.data());

    for (auto& fb: framebuffers) {
        vkDestroyFramebuffer(ctx.device, fb, ALLOCATOR);
    }
    depthImage.destroy(ctx.device);
    vkDestroyPipeline(ctx.device, pipeline, ALLOCATOR);
    vkDestroyPipelineLayout(ctx.device, pipelineLayout, ALLOCATOR);
    vkDestroyDescriptorSetLayout(ctx.device, descriptorSetLayout, ALLOCATOR);
    vkDestroyRenderPass(ctx.device, renderPass, ALLOCATOR);
}

void testCorrectness() {
    std::size_t N = 128;
    std::vector<Vertex> data(N);
    initParticleBuffer_ring_flat(data.data(), N);
    //initParticleBuffer_2_body(data.data(), N);

    auto diaplayAll = [&]() {
        for (int i = 0; i < N; i++) {
            std::cout << "   " << i << " pos: " << glm::to_string(data[i].pos_m) << '\n';
            std::cout << "   " << i << " vel: " << glm::to_string(data[i].velocity) << '\n';
        }
    };

    float refH = test_alg::computeHamiltonian(data.data(), N);
    for (auto i = 0; i <= 30000; i++) {
        if (i % 60 == 0) {
            std::cout << i << " sec: " << i/60 << " H = " <<
                         test_alg::computeHamiltonian(data.data(), N) << '\n';
            std::cout << i << " sec: " << i/60 << "  H diff = " <<
                         refH - test_alg::computeHamiltonian(data.data(), N) << '\n';
            //diaplayAll();
        }
        test_alg::update(data.data(), N);
    }
}


int main()
{
    std::cout << "Shaders dir: " << SHADERS_DIR << '\n';
    initWindow(runGraphics);
    //initWindow(runGraphics_test_star_shader);
    //initWindow(testCompute_simple);

    //testCorrectness();
    return 0;
}



