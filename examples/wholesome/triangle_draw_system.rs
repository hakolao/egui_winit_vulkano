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

use std::{convert::TryInto, sync::Arc};

use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, CommandBuffer, CommandBufferBeginInfo,
        CommandBufferInheritanceInfo, RecordingCommandBuffer,
    },
    device::Queue,
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            depth_stencil::DepthStencilState,
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    render_pass::Subpass,
};

use crate::renderer::Allocators;

pub struct TriangleDrawSystem {
    gfx_queue: Arc<Queue>,
    vertex_buffer: Subbuffer<[MyVertex]>,
    pipeline: Arc<GraphicsPipeline>,
    subpass: Subpass,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
}

impl TriangleDrawSystem {
    pub fn new(
        gfx_queue: Arc<Queue>,
        subpass: Subpass,
        allocators: &Allocators,
    ) -> TriangleDrawSystem {
        let vertex_buffer = Buffer::from_iter(
            allocators.memory.clone(),
            BufferCreateInfo { usage: BufferUsage::VERTEX_BUFFER, ..Default::default() },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            [
                MyVertex { position: [-0.5, -0.25], color: [1.0, 0.0, 0.0, 1.0] },
                MyVertex { position: [0.0, 0.5], color: [0.0, 1.0, 0.0, 1.0] },
                MyVertex { position: [0.25, -0.1], color: [0.0, 0.0, 1.0, 1.0] },
            ],
        )
        .unwrap();

        let pipeline = {
            let vs = vs::load(gfx_queue.device().clone())
                .expect("failed to create shader module")
                .entry_point("main")
                .unwrap();
            let fs = fs::load(gfx_queue.device().clone())
                .expect("failed to create shader module")
                .entry_point("main")
                .unwrap();

            let vertex_input_state =
                MyVertex::per_vertex().definition(&vs.info().input_interface).unwrap();

            let stages =
                [PipelineShaderStageCreateInfo::new(vs), PipelineShaderStageCreateInfo::new(fs)];

            let layout = PipelineLayout::new(
                gfx_queue.device().clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(gfx_queue.device().clone())
                    .unwrap(),
            )
            .unwrap();

            GraphicsPipeline::new(gfx_queue.device().clone(), None, GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState::default()),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),
                depth_stencil_state: Some(DepthStencilState::default()),
                dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                subpass: Some(subpass.clone().into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            })
            .unwrap()
        };

        TriangleDrawSystem {
            gfx_queue,
            vertex_buffer,
            pipeline,
            subpass,
            command_buffer_allocator: allocators.command_buffers.clone(),
        }
    }

    pub fn draw(&self, viewport_dimensions: [u32; 2]) -> Arc<CommandBuffer> {
        let mut builder = RecordingCommandBuffer::new(
            self.command_buffer_allocator.clone(),
            self.gfx_queue.queue_family_index(),
            vulkano::command_buffer::CommandBufferLevel::Secondary,
            CommandBufferBeginInfo {
                inheritance_info: Some(CommandBufferInheritanceInfo {
                    render_pass: Some(self.subpass.clone().into()),
                    ..Default::default()
                }),
                ..Default::default()
            },
        )
        .unwrap();
        builder
            .bind_pipeline_graphics(self.pipeline.clone())
            .unwrap()
            .set_viewport(
                0,
                [Viewport {
                    offset: [0.0, 0.0],
                    extent: [viewport_dimensions[0] as f32, viewport_dimensions[1] as f32],
                    depth_range: 0.0..=1.0,
                }]
                .into_iter()
                .collect(),
            )
            .unwrap()
            .bind_vertex_buffers(0, self.vertex_buffer.clone())
            .unwrap();
        unsafe {
            builder.draw(self.vertex_buffer.len() as u32, 1, 0, 0).unwrap();
        }
        builder.end().unwrap()
    }
}

#[repr(C)]
#[derive(BufferContents, Vertex)]
struct MyVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
    #[format(R32G32B32A32_SFLOAT)]
    color: [f32; 4],
}

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
