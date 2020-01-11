find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED IMPORTED_TARGET glfw3)

add_library(LibGLFW INTERFACE IMPORTED GLOBAL)
target_link_libraries(LibGLFW INTERFACE PkgConfig::GLFW)

# # Instead of alias, also possible to do this in this way:
# pkg_search_module(GLFW REQUIRED IMPORTED_TARGET GLOBAL glfw3)
# So pkg_search_module import as GLOBAL, and in the root target add:
# target_link_libraries(cmaketest PRIVATE PkgConfig::GLFW)
