#version 450

layout(location = 0) in vec4 inPos;
//layout(location = 1) in vec3 inVelocity;
//layout(location = 2) in float inWeight;

layout(location = 0) out vec3 fragColor;

// aspectRatio = width/height
layout(push_constant) uniform PushConstants { mat4 mvp; };

void main() {
    // (x, y * aspectRatio) = it means that screen widht = 2.0 (-1.0 .. 1.0)
    //vec4 pos = vec4(inPos.xyz, 1.0);
    // Visible region is -1..1 in on all axies
    //float z = 0.5*(inPos.z + 1.0); // Transform: -1..1 -> 0..1
    gl_Position = mvp * vec4(inPos.x, inPos.y, inPos.z, 1.0);


    // TODO: not gl_PointSize value is supported (check if vulkan documentation!) 
    //fragColor = vec3(inPos.w, 1.0 - inPos.w, 1.0 - inPos.w);

    //fragColor = vec3(1.0);

    /* Final
    float zc = 1.0 - clamp(gl_Position.z/gl_Position.w, 0.0, 1.0);
    // gl_PointSize = 10 + int(20.0 * zc); // for resolution {2980, 2000}
    gl_PointSize = 5 + int(10.0 * zc); // for resolution {1980, 1080}

    // zc* -> object farther is a bit darker
    fragColor = zc * vec3(210.0f/255.0, 236.0f/255.0, 1.0);
    */

    gl_PointSize = 30.0;
    fragColor = vec3(210.0f/255.0, 236.0f/255.0, 1.0);
}
