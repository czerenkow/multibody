#version 450

layout(location = 0) in vec4 inPos;
layout(location = 0) out vec3 fragColor;
layout(push_constant) uniform PushConstants { mat4 mvp; };

void main() {
    gl_Position = mvp * vec4(inPos.x, inPos.y, inPos.z, 1.0);

    // Final
    float zc = 1.0 - clamp(gl_Position.z/gl_Position.w, 0.0, 1.0);
    // TODO: Is gl_PointSize value supported? (check if vulkan documentation!) 
    gl_PointSize = 5 + int(10.0 * zc);     // good for resolution {1980, 1080}
    // gl_PointSize = 10 + int(20.0 * zc); // good for resolution {2980, 2000}

    // zc* -> object farther is a bit darker
    fragColor = zc * vec3(210.0f/255.0, 236.0f/255.0, 1.0);

    // For testing 
    // gl_PointSize = 30.0;
    // fragColor = vec3(210.0f/255.0, 236.0f/255.0, 1.0);
}
