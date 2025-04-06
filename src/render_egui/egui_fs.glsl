#version 450

#include "_push_constants.glsl"

layout(location = 0) in vec4 v_color;
layout(location = 1) in vec2 v_tex_coords;

layout(location = 0) out vec4 f_color;

// 0-1 sRGB  from  0-1 linear
vec3 srgb_from_linear(vec3 linear) {
    bvec3 cutoff = lessThan(linear, vec3(0.0031308));
    vec3 lower = linear * vec3(12.92);
    vec3 higher = vec3(1.055) * pow(linear, vec3(1./2.4)) - vec3(0.055);
    return mix(higher, lower, vec3(cutoff));
}

// 0-1 sRGBA  from  0-1 linear
vec4 srgba_from_linear(vec4 linear) {
    return vec4(srgb_from_linear(linear.rgb), linear.a);
}

// 0-1 linear  from  0-1 sRGB
vec3 linear_from_srgb(vec3 srgb) {
    bvec3 cutoff = lessThan(srgb, vec3(0.04045));
    vec3 lower = srgb / vec3(12.92);
    vec3 higher = pow((srgb + vec3(0.055) / vec3(1.055)), vec3(2.4));
    return mix(higher, lower, vec3(cutoff));
}

// 0-1 linear  from  0-1 sRGB
vec4 linear_from_srgba(vec4 srgb) {
    return vec4(linear_from_srgb(srgb.rgb), srgb.a);
}

void main() {
    // ALL calculations should be done in gamma space, this includes texture * color and blending
    vec4 texture_color = srgba_from_linear(texture(vko_sampler2D(texture_id, sampler_id), v_tex_coords));
    vec4 color = v_color * texture_color;

    // If output_in_linear_colorspace is true, we are rendering into an sRGB image, for which we'll convert to linear color space.
    // **This will break blending** as it will be performed in linear color space instead of sRGB like egui expects.
    if (output_in_linear_colorspace == 1) {
        color = linear_from_srgba(color);
    }
    f_color = color;
}