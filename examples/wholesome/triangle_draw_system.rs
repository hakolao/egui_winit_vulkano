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

use bytemuck::{Pod, Zeroable};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferInheritanceInfo, CommandBufferUsage,
        SecondaryAutoCommandBuffer,
    },
    device::Queue,
    pipeline::{
        graphics::{
            depth_stencil::DepthStencilState,
            input_assembly::InputAssemblyState,
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline,
    },
    render_pass::Subpass,
};

pub struct TriangleDrawSystem {
    gfx_queue: Arc<Queue>,
    vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
    pipeline: Arc<GraphicsPipeline>,
    subpass: Subpass,
}

impl TriangleDrawSystem {
    pub fn new(gfx_queue: Arc<Queue>, subpass: Subpass) -> TriangleDrawSystem {
        let vertex_buffer = {
            CpuAccessibleBuffer::from_iter(
                gfx_queue.device().clone(),
                BufferUsage { vertex_buffer: true, ..BufferUsage::empty() },
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
            let vs = vs::load(gfx_queue.device().clone()).expect("failed to create shader module");
            let fs = fs::load(gfx_queue.device().clone()).expect("failed to create shader module");

            GraphicsPipeline::start()
                .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
                .vertex_shader(vs.entry_point("main").unwrap(), ())
                .input_assembly_state(InputAssemblyState::new())
                .fragment_shader(fs.entry_point("main").unwrap(), ())
                .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
                .depth_stencil_state(DepthStencilState::simple_depth_test())
                .render_pass(subpass.clone())
                .build(gfx_queue.device().clone())
                .unwrap()
        };

        TriangleDrawSystem { gfx_queue, vertex_buffer, pipeline, subpass }
    }

    pub fn draw(&self, viewport_dimensions: [u32; 2]) -> SecondaryAutoCommandBuffer {
        let mut builder = AutoCommandBufferBuilder::secondary(
            self.gfx_queue.device().clone(),
            self.gfx_queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
            CommandBufferInheritanceInfo {
                render_pass: Some(self.subpass.clone().into()),
                ..Default::default()
            },
        )
        .unwrap();
        builder
            .bind_pipeline_graphics(self.pipeline.clone())
            .set_viewport(0, [Viewport {
                origin: [0.0, 0.0],
                dimensions: [viewport_dimensions[0] as f32, viewport_dimensions[1] as f32],
                depth_range: 0.0..1.0,
            }])
            .bind_vertex_buffers(0, self.vertex_buffer.clone())
            .draw(self.vertex_buffer.len() as u32, 1, 0, 0)
            .unwrap();
        builder.build().unwrap()
    }
}

#[repr(C)]
#[derive(Default, Debug, Copy, Clone, Zeroable, Pod)]
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
