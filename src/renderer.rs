// Copyright (c) 2021 Okko Hakola
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;

use egui::{paint::Mesh, Rect};
use vulkano::{
    buffer::{BufferAccess, BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, DynamicState, SubpassContents},
    descriptor::{
        descriptor_set::{PersistentDescriptorSet, UnsafeDescriptorSetLayout},
        DescriptorSet, PipelineLayoutAbstract,
    },
    device::Queue,
    format::{B8G8R8A8Unorm, Format},
    framebuffer::{Framebuffer, RenderPassAbstract, Subpass},
    image::{AttachmentImage, ImageAccess, ImageViewAccess},
    pipeline::{
        viewport::{Scissor, Viewport},
        GraphicsPipeline, GraphicsPipelineAbstract,
    },
    sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode},
    sync::GpuFuture,
};

use crate::{context::Context, utils::texture_from_bgra_bytes};

const VERTICES_PER_QUAD: usize = 4;
const VERTEX_BUFFER_SIZE: usize = 1024 * 1024 * VERTICES_PER_QUAD;
const INDEX_BUFFER_SIZE: usize = 1024 * 1024 * 2;

/// Should match vertex definition of egui (except color is `[f32; 4]`)
#[derive(Default, Debug, Clone, Copy)]
pub struct EguiVertex {
    pub position: [f32; 2],
    pub tex_coords: [f32; 2],
    pub color: [f32; 4],
}
vulkano::impl_vertex!(EguiVertex, position, tex_coords, color);

pub struct Renderer {
    gfx_queue: Arc<Queue>,
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,

    vertex_buffer: Arc<CpuAccessibleBuffer<[EguiVertex]>>,
    index_buffer: Arc<CpuAccessibleBuffer<[u32]>>,
    depth_buffer: Arc<AttachmentImage>,
    pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,

    egui_texture_version: u64,
    egui_texture_desc_set: Arc<dyn DescriptorSet + Send + Sync>,

    user_texture_desc_sets: Vec<Option<Arc<dyn DescriptorSet + Send + Sync>>>,
}

impl Renderer {
    /// Creates a new [EguiVulkanoRenderer] which is responsible for rendering egui
    /// See examples
    pub fn new(gfx_queue: Arc<Queue>, final_output_format: Format) -> Renderer {
        // Create Gui render pass with just depth and final color
        let render_pass = Arc::new(
            vulkano::ordered_passes_renderpass!(gfx_queue.device().clone(),
                attachments: {
                    final_color: {
                        load: Clear,
                        store: Store,
                        format: final_output_format,
                        samples: 1,
                    },
                    depth: {
                        load: Clear,
                        store: DontCare,
                        format: Format::D16Unorm,
                        samples: 1,
                    }
                },
                passes: [
                    {
                        color: [final_color],
                        depth_stencil: {depth},
                        input: []
                    }
                ]
            )
            .unwrap(),
        );
        let depth_buffer = AttachmentImage::transient_input_attachment(
            gfx_queue.device().clone(),
            [1, 1],
            Format::D16Unorm,
        )
        .unwrap();
        // Create vertex and index buffers
        let vertex_buffer = unsafe {
            CpuAccessibleBuffer::<[EguiVertex]>::uninitialized_array(
                gfx_queue.device().clone(),
                VERTEX_BUFFER_SIZE,
                BufferUsage::vertex_buffer(),
                false,
            )
            .expect("failed to create gui vertex buffer")
        };
        let index_buffer = unsafe {
            CpuAccessibleBuffer::<[u32]>::uninitialized_array(
                gfx_queue.device().clone(),
                INDEX_BUFFER_SIZE,
                BufferUsage::index_buffer(),
                false,
            )
            .expect("failed to create gui vertex buffer")
        };
        // Create pipeline
        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();
        let pipeline = {
            let vs = vs::Shader::load(gfx_queue.device().clone())
                .expect("failed to create shader module");
            let fs = fs::Shader::load(gfx_queue.device().clone())
                .expect("failed to create shader module");
            Arc::new(
                GraphicsPipeline::start()
                    .vertex_input_single_buffer::<EguiVertex>()
                    .vertex_shader(vs.main_entry_point(), ())
                    .triangle_list()
                    .fragment_shader(fs.main_entry_point(), ())
                    .viewports_scissors_dynamic(1)
                    .blend_alpha_blending()
                    .render_pass(subpass)
                    .build(gfx_queue.device().clone())
                    .unwrap(),
            )
        };

        // Create image attachments (temporary)
        let layout = pipeline.descriptor_set_layout(0).unwrap();
        // Create temp font image (gets replaced in draw)
        let font_image =
            AttachmentImage::sampled(gfx_queue.device().clone(), [1, 1], B8G8R8A8Unorm).unwrap();
        // Create font image desc set
        let font_desc_set =
            Self::sampled_image_desc_set(gfx_queue.clone(), layout, font_image.clone());
        Renderer {
            gfx_queue,
            render_pass,
            vertex_buffer,
            index_buffer,
            depth_buffer,
            pipeline,
            egui_texture_version: 0,
            egui_texture_desc_set: font_desc_set,
            user_texture_desc_sets: vec![],
        }
    }

    /// Creates a descriptor set for images
    fn sampled_image_desc_set(
        gfx_queue: Arc<Queue>,
        layout: &Arc<UnsafeDescriptorSetLayout>,
        image: Arc<dyn ImageViewAccess + Send + Sync>,
    ) -> Arc<dyn DescriptorSet + Send + Sync> {
        let sampler = Sampler::new(
            gfx_queue.device().clone(),
            Filter::Linear,
            Filter::Linear,
            MipmapMode::Linear,
            SamplerAddressMode::ClampToEdge,
            SamplerAddressMode::ClampToEdge,
            SamplerAddressMode::ClampToEdge,
            0.0,
            1.0,
            0.0,
            1000.0,
        )
        .expect("Failed to create sampler");
        Arc::new(
            PersistentDescriptorSet::start(layout.clone())
                .add_sampled_image(image.clone(), sampler.clone())
                .unwrap()
                .build()
                .expect("Failed to create descriptor set with sampler"),
        )
    }

    /// Registers a user texture. User texture needs to be unregistered when it is no longer needed
    pub fn register_user_image(
        &mut self,
        image: Arc<dyn ImageViewAccess + Send + Sync>,
    ) -> egui::TextureId {
        // get texture id, if one has been unregistered, give that id as new id
        let id = if let Some(i) = self.user_texture_desc_sets.iter().position(|utds| utds.is_none())
        {
            i as u64
        } else {
            self.user_texture_desc_sets.len() as u64
        };
        let layout = self.pipeline.descriptor_set_layout(0).unwrap();
        let desc_set = Self::sampled_image_desc_set(self.gfx_queue.clone(), layout, image);
        if id == self.user_texture_desc_sets.len() as u64 {
            self.user_texture_desc_sets.push(Some(desc_set));
        } else {
            self.user_texture_desc_sets[id as usize] = Some(desc_set);
        }
        egui::TextureId::User(id)
    }

    /// Unregister user texture.
    pub fn unregister_user_image(&mut self, texture_id: egui::TextureId) {
        if let egui::TextureId::User(id) = texture_id {
            if let Some(_descriptor_set) = self.user_texture_desc_sets[id as usize].as_ref() {
                self.user_texture_desc_sets[id as usize] = None;
            }
        }
    }

    fn update_font_texture(&mut self, egui_context: &Context) {
        let texture = egui_context.context().texture();
        if texture.version == self.egui_texture_version {
            return;
        }
        let data = texture.pixels.iter().flat_map(|&r| vec![r, r, r, r]).collect::<Vec<_>>();
        // Update font image
        let font_image = texture_from_bgra_bytes(
            self.gfx_queue.clone(),
            &data,
            (texture.width as u64, texture.height as u64),
        )
        .expect("Failed to load font image");
        self.egui_texture_version = texture.version;
        // Update descriptor set
        let layout = self.pipeline.descriptor_set_layout(0).unwrap();
        let font_desc_set =
            Self::sampled_image_desc_set(self.gfx_queue.clone(), layout, font_image.clone());
        self.egui_texture_desc_set = font_desc_set;
    }

    fn get_rect_scissor(
        &self,
        egui_context: &mut Context,
        framebuffer_dimensions: [u32; 2],
        rect: Rect,
    ) -> Scissor {
        let min = rect.min;
        let min = egui::Pos2 {
            x: min.x * egui_context.scale_factor() as f32,
            y: min.y * egui_context.scale_factor() as f32,
        };
        let min = egui::Pos2 {
            x: egui::math::clamp(min.x, 0.0..=framebuffer_dimensions[0] as f32),
            y: egui::math::clamp(min.y, 0.0..=framebuffer_dimensions[1] as f32),
        };
        let max = rect.max;
        let max = egui::Pos2 {
            x: max.x * egui_context.scale_factor() as f32,
            y: max.y * egui_context.scale_factor() as f32,
        };
        let max = egui::Pos2 {
            x: egui::math::clamp(max.x, min.x..=framebuffer_dimensions[0] as f32),
            y: egui::math::clamp(max.y, min.y..=framebuffer_dimensions[1] as f32),
        };
        Scissor {
            origin: [min.x.round() as i32, min.y.round() as i32],
            dimensions: [(max.x.round() - min.x) as u32, (max.y.round() - min.y) as u32],
        }
    }

    fn resize_allocations(&mut self, new_vertices_size: usize, new_indices_size: usize) {
        let vertex_buffer = unsafe {
            CpuAccessibleBuffer::<[EguiVertex]>::uninitialized_array(
                self.gfx_queue.device().clone(),
                new_vertices_size,
                BufferUsage::vertex_buffer(),
                false,
            )
            .expect("failed to create gui vertex buffer")
        };
        let index_buffer = unsafe {
            CpuAccessibleBuffer::<[u32]>::uninitialized_array(
                self.gfx_queue.device().clone(),
                new_indices_size,
                BufferUsage::index_buffer(),
                false,
            )
            .expect("failed to create gui vertex buffer")
        };
        self.vertex_buffer = vertex_buffer;
        self.index_buffer = index_buffer;
    }

    fn copy_mesh(&self, mesh: Mesh, vertex_start: usize, index_start: usize) {
        // Copy vertices to buffer
        let v_slice = &mesh.vertices;
        let mut vertex_content = self.vertex_buffer.write().unwrap();
        let mut slice_i = 0;
        for i in vertex_start..(vertex_start + v_slice.len()) {
            let v = v_slice[slice_i];
            vertex_content[i] = EguiVertex {
                position: [v.pos.x, v.pos.y],
                tex_coords: [v.uv.x, v.uv.y],
                color: [
                    v.color.r() as f32 / 255.0,
                    v.color.g() as f32 / 255.0,
                    v.color.b() as f32 / 255.0,
                    v.color.a() as f32 / 255.0,
                ],
            };
            slice_i += 1;
        }
        // Copy indices to buffer
        let i_slice = &mesh.indices;
        let mut index_content = self.index_buffer.write().unwrap();
        slice_i = 0;
        for i in index_start..(index_start + i_slice.len()) {
            let index = i_slice[slice_i];
            index_content[i] = index;
            slice_i += 1;
        }
    }

    fn resize_needed(&self, vertex_end: usize, index_end: usize) -> bool {
        let vtx_size = std::mem::size_of::<EguiVertex>();
        let idx_size = std::mem::size_of::<u32>();
        vertex_end * vtx_size >= self.vertex_buffer.size()
            || index_end * idx_size >= self.index_buffer.size()
    }

    // Starts the rendering pipeline and returns [`AutoCommandBufferBuilder`] for drawing
    fn start<I>(&mut self, final_image: I) -> (AutoCommandBufferBuilder, [u32; 2])
    where
        I: ImageAccess + ImageViewAccess + Clone + Send + Sync + 'static,
    {
        // Get dimensions
        let img_dims = ImageAccess::dimensions(&final_image).width_height();
        // Resize depth attachment to match final image's dimensions
        if ImageAccess::dimensions(&self.depth_buffer).width_height() != img_dims {
            self.depth_buffer = AttachmentImage::transient_input_attachment(
                self.gfx_queue.device().clone(),
                img_dims,
                Format::D16Unorm,
            )
            .unwrap();
        }
        // Create framebuffer (must be in same order as render pass description in `new`
        let framebuffer = Arc::new(
            Framebuffer::start(self.render_pass.clone())
                .add(final_image.clone())
                .unwrap()
                .add(self.depth_buffer.clone())
                .unwrap()
                .build()
                .unwrap(),
        );
        let mut command_buffer_builder = AutoCommandBufferBuilder::primary_one_time_submit(
            self.gfx_queue.device().clone(),
            self.gfx_queue.family(),
        )
        .unwrap();
        // Add clear values here for attachments and begin render pass
        command_buffer_builder
            .begin_render_pass(framebuffer.clone(), SubpassContents::SecondaryCommandBuffers, vec![
                [0.0, 0.0, 0.0, 0.0].into(),
                1.0f32.into(),
            ])
            .unwrap();
        (command_buffer_builder, img_dims)
    }

    /// Executes our draw commands on the final image and returns a `GpuFuture` to wait on
    pub fn draw<F, I>(
        &mut self,
        egui_context: &mut Context,
        clipped_meshes: Vec<egui::ClippedMesh>,
        before_future: F,
        final_image: I,
    ) -> Box<dyn GpuFuture>
    where
        F: GpuFuture + 'static,
        I: ImageAccess + ImageViewAccess + Clone + Send + Sync + 'static,
    {
        let (mut command_buffer_builder, framebuffer_dimensions) = self.start(final_image);
        egui_context.update_elapsed_time();
        self.update_font_texture(egui_context);
        let push_constants = vs::ty::PushConstants {
            screen_size: [
                framebuffer_dimensions[0] as f32 / egui_context.scale_factor() as f32,
                framebuffer_dimensions[1] as f32 / egui_context.scale_factor() as f32,
            ],
        };
        let mut builder = AutoCommandBufferBuilder::secondary_graphics(
            self.gfx_queue.device().clone(),
            self.gfx_queue.family(),
            self.pipeline.clone().subpass(),
        )
        .unwrap();

        let mut vertex_start = 0;
        let mut index_start = 0;
        for egui::ClippedMesh(rect, mesh) in clipped_meshes {
            // Nothing to draw if we don't have vertices & indices
            if mesh.vertices.is_empty() || mesh.indices.is_empty() {
                continue;
            }
            let mut user_image_id = None;
            if let egui::TextureId::User(id) = mesh.texture_id {
                // No user image available anymore, don't draw
                if self.user_texture_desc_sets[id as usize].is_none() {
                    eprintln!("This user texture no longer exists {:?}", mesh.texture_id);
                    continue;
                }
                user_image_id = Some(id);
            }

            let scissors = vec![self.get_rect_scissor(egui_context, framebuffer_dimensions, rect)];
            let dynamic_state = DynamicState {
                viewports: Some(vec![Viewport {
                    origin: [0.0, 0.0],
                    dimensions: [
                        framebuffer_dimensions[0] as f32,
                        framebuffer_dimensions[1] as f32,
                    ],
                    depth_range: 0.0..1.0,
                }]),
                scissors: Some(scissors),
                ..DynamicState::none()
            };
            let vertices_count = mesh.vertices.len();
            let indices_count = mesh.indices.len();
            // Resize buffers if needed
            if self.resize_needed(vertex_start + vertices_count, index_start + indices_count) {
                self.resize_allocations(
                    self.vertex_buffer.size() * 2,
                    self.index_buffer.size() * 2,
                );
                // Stop copying and continue next frame
                break;
            }
            self.copy_mesh(mesh, vertex_start, index_start);
            // Access vertex & index slices for drawing
            let vertices = Arc::new(
                self.vertex_buffer
                    .clone()
                    .into_buffer_slice()
                    .slice(vertex_start..(vertex_start + vertices_count))
                    .unwrap(),
            );
            let indices = Arc::new(
                self.index_buffer
                    .clone()
                    .into_buffer_slice()
                    .slice(index_start..(index_start + indices_count))
                    .unwrap(),
            );
            let desc_set = if let Some(id) = user_image_id {
                self.user_texture_desc_sets[id as usize].as_ref().unwrap().clone()
            } else {
                self.egui_texture_desc_set.clone()
            };
            builder
                .draw_indexed(
                    self.pipeline.clone(),
                    &dynamic_state,
                    vec![vertices.clone()],
                    indices.clone(),
                    desc_set,
                    push_constants,
                    vec![],
                )
                .unwrap();
            vertex_start += vertices_count;
            index_start += indices_count;
        }
        // Execute draw commands
        let command_buffer = builder.build().unwrap();
        unsafe {
            command_buffer_builder.execute_commands(command_buffer).unwrap();
        }
        self.finish(command_buffer_builder, Box::new(before_future))
    }

    // Finishes the rendering pipeline
    fn finish(
        &self,
        mut command_buffer_builder: AutoCommandBufferBuilder,
        before_main_cb_future: Box<dyn GpuFuture>,
    ) -> Box<dyn GpuFuture> {
        // We end render pass
        command_buffer_builder.end_render_pass().unwrap();
        // Then execute our whole command buffer
        let command_buffer = command_buffer_builder.build().unwrap();
        let after_main_cb =
            before_main_cb_future.then_execute(self.gfx_queue.clone(), command_buffer).unwrap();
        let future =
            after_main_cb.then_signal_fence_and_flush().expect("Failed to signal fence and flush");
        // Return our future
        Box::new(future)
    }

    pub fn queue(&self) -> Arc<Queue> {
        self.gfx_queue.clone()
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
#version 450

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 tex_coords;
layout(location = 2) in vec4 color;

layout(location = 0) out vec4 v_color;
layout(location = 1) out vec2 v_tex_coords;

layout(push_constant) uniform PushConstants {
    vec2 screen_size;
} push_constants;

void main() {
  gl_Position =
      vec4(2.0 * position.x / push_constants.screen_size.x - 1.0,
           2.0 * position.y / push_constants.screen_size.y - 1.0, 0.0, 1.0);
  v_color = color;
  v_tex_coords = tex_coords;
}"
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
#version 450

layout(location = 0) in vec4 v_color;
layout(location = 1) in vec2 v_tex_coords;

layout(location = 0) out vec4 f_color;

layout(binding = 0, set = 0) uniform sampler2D font_texture;

void main() {
    f_color = v_color * texture(font_texture, v_tex_coords);
}"
    }
}
