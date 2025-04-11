#version 450

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 tex_coords;
layout(location = 2) in vec4 color;

layout(location = 0) out vec4 v_color;
layout(location = 1) out vec2 v_tex_coords;

#include "_push_constants.glsl"

void main() {
    gl_Position = vec4(
        2.0 * position.x / screen_size.x - 1.0,
        2.0 * position.y / screen_size.y - 1.0,
        0.0, 1.0
    );
    v_color = color;
    v_tex_coords = tex_coords;
}