find_package(Vulkan REQUIRED)

# Make found target globally available.
# if (Vulkan_FOUND) ...
set_target_properties(Vulkan::Vulkan PROPERTIES IMPORTED_GLOBAL TRUE)
