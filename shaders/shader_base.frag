#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 outColor;

void main() {
    vec2 v = 2.0 * (gl_PointCoord - vec2(0.5, 0.5));
    //vec3 c = vec3(fragColor.x * gl_PointCoord.x, fragColor.y * gl_PointCoord.y, 0.0);
    float vlen1 = clamp(1.0 - length(v), 0.0, 1.0);
    vlen1 = vlen1 * vlen1;
    //outColor = vec4(1.0, 1.0, 1.0, 0.2);
    //outColor = vec4(fragColor * vlen1, 0.0);
    //outColor = vec4(fragColor, 0.5);
    outColor.rgb = fragColor * vlen1; //vec4(fragColor);
}
