// Copyright (c) 2017 The vulkano developers <=== !
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

// Slightly modified version from
// https://github.com/vulkano-rs/vulkano-examples/blob/master/src/bin/deferred/triangle_draw_system.rs
// To simplify this wholesome example :)

use std::sync::Arc;

use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBuffer, AutoCommandBufferBuilder, DynamicState},
    device::Queue,
    framebuffer::{RenderPassAbstract, Subpass},
    pipeline::{viewport::Viewport, GraphicsPipeline, GraphicsPipelineAbstract},
};

pub struct TriangleDrawSystem {
    gfx_queue: Arc<Queue>,
    vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
    pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
}

impl TriangleDrawSystem {
    pub fn new<R>(gfx_queue: Arc<Queue>, subpass: Subpass<R>) -> TriangleDrawSystem
    where
        R: RenderPassAbstract + Send + Sync + 'static,
    {
        let vertex_buffer = {
            CpuAccessibleBuffer::from_iter(
                gfx_queue.device().clone(),
                BufferUsage::all(),
                false,
                [
                    Vertex { position: [-0.5, -0.25], color: [1.0, 0.0, 0.0, 1.0] },
                    Vertex { position: [0.0, 0.5], color: [0.0, 1.0, 0.0, 1.0] },
                    Vertex { position: [0.25, -0.1], color: [0.0, 0.0, 1.0, 1.0] },
                ]
                .iter()
                .cloned(),
            )
            .expect("failed to create buffer")
        };
        let pipeline = {
            let vs = vs::Shader::load(gfx_queue.device().clone())
                .expect("failed to create shader module");
            let fs = fs::Shader::load(gfx_queue.device().clone())
                .expect("failed to create shader module");

            Arc::new(
                GraphicsPipeline::start()
                    .vertex_input_single_buffer::<Vertex>()
                    .vertex_shader(vs.main_entry_point(), ())
                    .triangle_list()
                    .viewports_dynamic_scissors_irrelevant(1)
                    .fragment_shader(fs.main_entry_point(), ())
                    .depth_stencil_simple_depth()
                    .render_pass(subpass)
                    .build(gfx_queue.device().clone())
                    .unwrap(),
            ) as Arc<_>
        };

        TriangleDrawSystem { gfx_queue, vertex_buffer, pipeline }
    }

    pub fn draw(&self, viewport_dimensions: [u32; 2]) -> AutoCommandBuffer {
        let mut builder = AutoCommandBufferBuilder::secondary_graphics(
            self.gfx_queue.device().clone(),
            self.gfx_queue.family(),
            self.pipeline.clone().subpass(),
        )
        .unwrap();
        builder
            .draw(
                self.pipeline.clone(),
                &DynamicState {
                    viewports: Some(vec![Viewport {
                        origin: [0.0, 0.0],
                        dimensions: [viewport_dimensions[0] as f32, viewport_dimensions[1] as f32],
                        depth_range: 0.0..1.0,
                    }]),
                    ..DynamicState::none()
                },
                vec![self.vertex_buffer.clone()],
                (),
                (),
            )
            .unwrap();
        builder.build().unwrap()
    }
}

#[derive(Default, Debug, Clone)]
struct Vertex {
    position: [f32; 2],
    color: [f32; 4],
}
vulkano::impl_vertex!(Vertex, position, color);

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
#version 450
layout(location = 0) in vec2 position;
layout(location = 1) in vec4 color;

layout(location = 0) out vec4 v_color;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    v_color = color;
}"
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
#version 450
layout(location = 0) in vec4 v_color;

layout(location = 0) out vec4 f_color;

void main() {
    f_color = v_color;
}"
    }
}
