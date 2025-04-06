#include <vulkano.glsl>

layout(push_constant) uniform PushConstants {
    SampledImageId texture_id;
    SamplerId sampler_id;
    vec2 screen_size;
    int output_in_linear_colorspace;
};