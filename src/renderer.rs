// Copyright (c) 2021 Okko Hakola
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;

use ahash::AHashMap;
use egui::{epaint::Primitive, ClippedPrimitive, PaintCallbackInfo, Rect, TexturesDelta};
use vulkano::{
    buffer::{
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
        Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer,
    },
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, BufferImageCopy,
        CommandBufferInheritanceInfo, CommandBufferUsage, CopyBufferToImageInfo,
        PrimaryAutoCommandBuffer, RenderPassBeginInfo, SecondaryAutoCommandBuffer,
        SubpassBeginInfo, SubpassContents,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, layout::DescriptorSetLayout, DescriptorSet,
        WriteDescriptorSet,
    },
    device::Queue,
    format::{Format, NumericFormat},
    image::{
        sampler::{
            ComponentMapping, ComponentSwizzle, Filter, Sampler, SamplerAddressMode,
            SamplerCreateInfo, SamplerMipmapMode,
        },
        view::{ImageView, ImageViewCreateInfo},
        Image, ImageAspects, ImageCreateInfo, ImageLayout, ImageSubresourceLayers, ImageType,
        ImageUsage, SampleCount,
    },
    memory::{
        allocator::{
            AllocationCreateInfo, DeviceLayout, MemoryTypeFilter, StandardMemoryAllocator,
        },
        DeviceAlignment,
    },
    pipeline::{
        graphics::{
            color_blend::{
                AttachmentBlend, BlendFactor, ColorBlendAttachmentState, ColorBlendState,
            },
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Scissor, Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    sync::GpuFuture,
    DeviceSize, NonZeroDeviceSize,
};

use crate::utils::Allocators;

const VERTICES_PER_QUAD: DeviceSize = 4;
const VERTEX_BUFFER_SIZE: DeviceSize = 1024 * 1024 * VERTICES_PER_QUAD;
const INDEX_BUFFER_SIZE: DeviceSize = 1024 * 1024 * 2;

type VertexBuffer = Subbuffer<[egui::epaint::Vertex]>;
type IndexBuffer = Subbuffer<[u32]>;

/// Should match vertex definition of egui
#[repr(C)]
#[derive(BufferContents, Vertex)]
pub struct EguiVertex {
    #[format(R32G32_SFLOAT)]
    pub position: [f32; 2],
    #[format(R32G32_SFLOAT)]
    pub tex_coords: [f32; 2],
    #[format(R8G8B8A8_UNORM)]
    pub color: [u8; 4],
}

pub struct Renderer {
    gfx_queue: Arc<Queue>,
    render_pass: Option<Arc<RenderPass>>,
    is_overlay: bool,
    output_in_linear_colorspace: bool,

    #[allow(unused)]
    format: vulkano::format::Format,
    font_sampler: Arc<Sampler>,
    // May be R8G8_UNORM or R8G8B8A8_SRGB
    font_format: Format,

    allocators: Allocators,
    vertex_index_buffer_pool: SubbufferAllocator,
    pipeline: Arc<GraphicsPipeline>,
    subpass: Subpass,

    texture_desc_sets: AHashMap<egui::TextureId, Arc<DescriptorSet>>,
    texture_images: AHashMap<egui::TextureId, Arc<ImageView>>,
    next_native_tex_id: u64,
}

impl Renderer {
    pub fn new_with_subpass(
        gfx_queue: Arc<Queue>,
        final_output_format: Format,
        subpass: Subpass,
    ) -> Renderer {
        Self::new_internal(gfx_queue, final_output_format, subpass, None, false)
    }

    /// Creates a new [Renderer] which is responsible for rendering egui with its own renderpass
    /// See examples
    pub fn new_with_render_pass(
        gfx_queue: Arc<Queue>,
        final_output_format: Format,
        is_overlay: bool,
        samples: SampleCount,
    ) -> Renderer {
        // Create Gui render pass with just depth and final color
        let render_pass = if is_overlay {
            vulkano::single_pass_renderpass!(gfx_queue.device().clone(),
                attachments: {
                    final_color: {
                        format: final_output_format,
                        samples: samples,
                        load_op: Load,
                        store_op: Store,
                    }
                },
                pass: {
                        color: [final_color],
                        depth_stencil: {}
                }
            )
            .unwrap()
        } else {
            vulkano::single_pass_renderpass!(gfx_queue.device().clone(),
                attachments: {
                    final_color: {
                        format: final_output_format,
                        samples: samples,
                        load_op: Clear,
                        store_op: Store,
                    }
                },
                pass: {
                        color: [final_color],
                        depth_stencil: {}
                }
            )
            .unwrap()
        };
        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();
        Self::new_internal(gfx_queue, final_output_format, subpass, Some(render_pass), is_overlay)
    }

    fn new_internal(
        gfx_queue: Arc<Queue>,
        final_output_format: Format,
        subpass: Subpass,
        render_pass: Option<Arc<RenderPass>>,
        is_overlay: bool,
    ) -> Renderer {
        let output_in_linear_colorspace =
            // final_output_format.type_color().unwrap() == NumericType::SRGB;
            final_output_format.numeric_format_color().unwrap() == NumericFormat::SRGB;
        let allocators = Allocators::new_default(gfx_queue.device());
        let vertex_index_buffer_pool =
            SubbufferAllocator::new(allocators.memory.clone(), SubbufferAllocatorCreateInfo {
                arena_size: INDEX_BUFFER_SIZE + VERTEX_BUFFER_SIZE,
                buffer_usage: BufferUsage::INDEX_BUFFER | BufferUsage::VERTEX_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            });
        let pipeline = Self::create_pipeline(gfx_queue.clone(), subpass.clone());
        let font_sampler = Sampler::new(gfx_queue.device().clone(), SamplerCreateInfo {
            mag_filter: Filter::Linear,
            min_filter: Filter::Linear,
            address_mode: [SamplerAddressMode::ClampToEdge; 3],
            mipmap_mode: SamplerMipmapMode::Linear,
            ..Default::default()
        })
        .unwrap();
        let font_format = Self::choose_font_format(gfx_queue.device());
        Renderer {
            gfx_queue,
            format: final_output_format,
            render_pass,
            vertex_index_buffer_pool,
            pipeline,
            subpass,
            texture_desc_sets: AHashMap::default(),
            texture_images: AHashMap::default(),
            next_native_tex_id: 0,
            is_overlay,
            output_in_linear_colorspace,
            font_sampler,
            font_format,
            allocators,
        }
    }

    pub fn has_renderpass(&self) -> bool {
        self.render_pass.is_some()
    }

    fn create_pipeline(gfx_queue: Arc<Queue>, subpass: Subpass) -> Arc<GraphicsPipeline> {
        let vs = vs::load(gfx_queue.device().clone())
            .expect("failed to create shader module")
            .entry_point("main")
            .unwrap();
        let fs = fs::load(gfx_queue.device().clone())
            .expect("failed to create shader module")
            .entry_point("main")
            .unwrap();

        let mut blend = AttachmentBlend::alpha();
        blend.src_color_blend_factor = BlendFactor::One;
        blend.src_alpha_blend_factor = BlendFactor::OneMinusDstAlpha;
        blend.dst_alpha_blend_factor = BlendFactor::One;
        let blend_state = ColorBlendState {
            attachments: vec![ColorBlendAttachmentState {
                blend: Some(blend),
                ..Default::default()
            }],
            ..ColorBlendState::default()
        };

        let vertex_input_state = Some(EguiVertex::per_vertex().definition(&vs).unwrap());

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
            vertex_input_state,
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState::default()),
            rasterization_state: Some(RasterizationState::default()),
            multisample_state: Some(MultisampleState {
                rasterization_samples: subpass.num_samples().unwrap_or(SampleCount::Sample1),
                ..Default::default()
            }),
            color_blend_state: Some(blend_state),
            dynamic_state: [DynamicState::Viewport, DynamicState::Scissor].into_iter().collect(),
            subpass: Some(subpass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        })
        .unwrap()
    }

    /// Creates a descriptor set for images
    fn sampled_image_desc_set(
        &self,
        layout: &Arc<DescriptorSetLayout>,
        image: Arc<ImageView>,
        sampler: Arc<Sampler>,
    ) -> Arc<DescriptorSet> {
        DescriptorSet::new(
            self.allocators.descriptor_set.clone(),
            layout.clone(),
            [WriteDescriptorSet::image_view_sampler(0, image, sampler)],
            [],
        )
        .unwrap()
    }

    /// Registers a user texture. User texture needs to be unregistered when it is no longer needed
    pub fn register_image(
        &mut self,
        image: Arc<ImageView>,
        sampler_create_info: SamplerCreateInfo,
    ) -> egui::TextureId {
        let layout = self.pipeline.layout().set_layouts().first().unwrap();
        let sampler = Sampler::new(self.gfx_queue.device().clone(), sampler_create_info).unwrap();
        let desc_set = self.sampled_image_desc_set(layout, image.clone(), sampler);
        let id = egui::TextureId::User(self.next_native_tex_id);
        self.next_native_tex_id += 1;
        self.texture_desc_sets.insert(id, desc_set);
        self.texture_images.insert(id, image);
        id
    }

    /// Unregister user texture.
    pub fn unregister_image(&mut self, texture_id: egui::TextureId) {
        self.texture_desc_sets.remove(&texture_id);
        self.texture_images.remove(&texture_id);
    }
    /// Choose a font format, attempt to minimize memory footprint and CPU unpacking time
    /// by choosing a swizzled linear format.
    fn choose_font_format(device: &vulkano::device::Device) -> Format {
        // Some portability subset devices are unable to swizzle views.
        let supports_swizzle =
            !device.physical_device().supported_extensions().khr_portability_subset
                || device.physical_device().supported_features().image_view_format_swizzle;
        // Check that this format is supported for all our uses:
        let is_supported = |device: &vulkano::device::Device, format: Format| {
            device
                .physical_device()
                .image_format_properties(vulkano::image::ImageFormatInfo {
                    format,
                    usage: ImageUsage::SAMPLED
                        | ImageUsage::TRANSFER_DST
                        | ImageUsage::TRANSFER_SRC,
                    ..Default::default()
                })
                // Ok(Some(..)) is supported format for this usage.
                .is_ok_and(|properties| properties.is_some())
        };
        if supports_swizzle && is_supported(device, Format::R8G8_UNORM) {
            // We can save mem by swizzling in hardware!
            Format::R8G8_UNORM
        } else {
            // Rest of implementation assumes R8G8B8A8_SRGB anyway!
            Format::R8G8B8A8_SRGB
        }
    }
    /// Based on self.font_format, extract into bytes.
    fn pack_font_data_into(&self, data: &egui::FontImage, into: &mut [u8]) {
        match self.font_format {
            Format::R8G8_UNORM => {
                // Egui expects RGB to be linear in shader, but alpha to be *nonlinear.*
                // Thus, we use R channel for linear coverage, G for the same coverage converted to nonlinear.
                // Then gets swizzled up to RRRG to match expected values.
                let linear =
                    data.pixels.iter().map(|f| (f.clamp(0.0, 1.0 - f32::EPSILON) * 256.0) as u8);
                let bytes = linear
                    .zip(data.srgba_pixels(None))
                    .flat_map(|(linear, srgb)| [linear, srgb.a()]);

                into.iter_mut().zip(bytes).for_each(|(into, from)| *into = from);
            }
            Format::R8G8B8A8_SRGB => {
                // No special tricks, pack them directly.
                let bytes = data.srgba_pixels(None).flat_map(|color| color.to_array());
                into.iter_mut().zip(bytes).for_each(|(into, from)| *into = from);
            }
            // This is the exhaustive list of choosable font formats.
            _ => unreachable!(),
        }
    }
    fn image_size_bytes(&self, delta: &egui::epaint::ImageDelta) -> usize {
        match &delta.image {
            egui::ImageData::Color(c) => {
                // Always four bytes per pixel for sRGBA
                c.width() * c.height() * 4
            }
            egui::ImageData::Font(f) => {
                f.width()
                    * f.height()
                    * match self.font_format {
                        Format::R8G8_UNORM => 2,
                        Format::R8G8B8A8_SRGB => 4,
                        // Exhaustive list of valid font formats
                        _ => unreachable!(),
                    }
            }
        }
    }
    /// Write a single texture delta using the provided staging region and commandbuffer
    fn update_texture_within(
        &mut self,
        id: egui::TextureId,
        delta: &egui::epaint::ImageDelta,
        stage: Subbuffer<[u8]>,
        mapped_stage: &mut [u8],
        cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) {
        // Extract pixel data from egui, writing into our region of the stage buffer.
        let format = match &delta.image {
            egui::ImageData::Color(image) => {
                assert_eq!(
                    image.width() * image.height(),
                    image.pixels.len(),
                    "Mismatch between texture size and texel count"
                );
                let bytes = image.pixels.iter().flat_map(|color| color.to_array());
                mapped_stage.iter_mut().zip(bytes).for_each(|(into, from)| *into = from);
                Format::R8G8B8A8_SRGB
            }
            egui::ImageData::Font(image) => {
                // Dynamically pack based on chosen format
                self.pack_font_data_into(image, mapped_stage);
                self.font_format
            }
        };

        // Copy texture data to existing image if delta pos exists (e.g. font changed)
        if let Some(pos) = delta.pos {
            let Some(existing_image) = self.texture_images.get(&id) else {
                // Egui wants us to update this texture but we don't have it to begin with!
                panic!("attempt to write into non-existing image");
            };
            // Make sure delta image type and destination image type match.
            assert_eq!(existing_image.format(), format);

            // Defer upload of data
            cbb.copy_buffer_to_image(CopyBufferToImageInfo {
                regions: [BufferImageCopy {
                    // Buffer offsets are derived
                    image_offset: [pos[0] as u32, pos[1] as u32, 0],
                    image_extent: [delta.image.width() as u32, delta.image.height() as u32, 1],
                    // Always use the whole image (no arrays or mips are performed)
                    image_subresource: ImageSubresourceLayers {
                        aspects: ImageAspects::COLOR,
                        mip_level: 0,
                        array_layers: 0..1,
                    },
                    ..Default::default()
                }]
                .into(),
                ..CopyBufferToImageInfo::buffer_image(stage, existing_image.image().clone())
            })
            .unwrap();
        } else {
            // Otherwise save the newly created image
            let img = {
                let extent = [delta.image.width() as u32, delta.image.height() as u32, 1];
                Image::new(
                    self.allocators.memory.clone(),
                    ImageCreateInfo {
                        image_type: ImageType::Dim2d,
                        format,
                        extent,
                        usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                        initial_layout: ImageLayout::Undefined,
                        ..Default::default()
                    },
                    AllocationCreateInfo::default(),
                )
                .unwrap()
            };
            // Defer upload of data
            cbb.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(stage, img.clone()))
                .unwrap();
            // Swizzle packed font images up to a full premul white.
            let component_mapping = match format {
                Format::R8G8_UNORM => ComponentMapping {
                    r: ComponentSwizzle::Red,
                    g: ComponentSwizzle::Red,
                    b: ComponentSwizzle::Red,
                    a: ComponentSwizzle::Green,
                },
                _ => ComponentMapping::identity(),
            };
            let view = ImageView::new(img.clone(), ImageViewCreateInfo {
                component_mapping,
                ..ImageViewCreateInfo::from_image(&img)
            })
            .unwrap();
            // Create a descriptor for it
            let layout = self.pipeline.layout().set_layouts().first().unwrap();
            let desc_set =
                self.sampled_image_desc_set(layout, view.clone(), self.font_sampler.clone());
            // Save!
            self.texture_desc_sets.insert(id, desc_set);
            self.texture_images.insert(id, view);
        };
    }
    /// Write the entire texture delta for this frame.
    fn update_textures(&mut self, sets: &[(egui::TextureId, egui::epaint::ImageDelta)]) {
        // Allocate enough memory to upload every delta at once.
        let total_size_bytes =
            sets.iter().map(|(_, set)| self.image_size_bytes(set)).sum::<usize>() * 4;
        // Infallible - unless we're on a 128 bit machine? :P
        let total_size_bytes = u64::try_from(total_size_bytes).unwrap();
        let Ok(total_size_bytes) = vulkano::NonZeroDeviceSize::try_from(total_size_bytes) else {
            // Nothing to upload!
            return;
        };
        let buffer = Buffer::new(
            self.allocators.memory.clone(),
            BufferCreateInfo { usage: BufferUsage::TRANSFER_SRC, ..Default::default() },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            // Bytes, align of one, infallible.
            DeviceLayout::new(total_size_bytes, DeviceAlignment::MIN).unwrap(),
        )
        .unwrap();
        let buffer = Subbuffer::new(buffer);

        // Shared command buffer for every upload in this batch.
        let mut cbb = AutoCommandBufferBuilder::primary(
            self.allocators.command_buffer.clone(),
            self.gfx_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        {
            // Scoped to keep writer lock bounded
            // Should be infallible - Just made the buffer so it's exclusive, and we have host access to it.
            let mut writer = buffer.write().unwrap();

            // Keep track of where to write the next image to into the staging buffer.
            let mut past_buffer_end = 0usize;

            for (id, delta) in sets {
                let image_size_bytes = self.image_size_bytes(delta);
                let range = past_buffer_end..(image_size_bytes + past_buffer_end);

                // Bump for next loop
                past_buffer_end += image_size_bytes;

                // Represents the same memory in two ways. Writable memmap, and gpu-side description.
                let stage = buffer.clone().slice(range.start as u64..range.end as u64);
                let mapped_stage = &mut writer[range];

                self.update_texture_within(*id, delta, stage, mapped_stage, &mut cbb);
            }
        }

        // Execute every upload at once and await:
        let command_buffer = cbb.build().unwrap();
        // Executing on the graphics queue not only since it's what we have, but
        // we must guarantee a transfer granularity of [1,1,x] which graphics queue is required to have.
        command_buffer
            .execute(self.gfx_queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
    }

    fn get_rect_scissor(
        &self,
        scale_factor: f32,
        framebuffer_dimensions: [u32; 2],
        rect: Rect,
    ) -> Scissor {
        let min = rect.min;
        let min = egui::Pos2 { x: min.x * scale_factor, y: min.y * scale_factor };
        let min = egui::Pos2 {
            x: min.x.clamp(0.0, framebuffer_dimensions[0] as f32),
            y: min.y.clamp(0.0, framebuffer_dimensions[1] as f32),
        };
        let max = rect.max;
        let max = egui::Pos2 { x: max.x * scale_factor, y: max.y * scale_factor };
        let max = egui::Pos2 {
            x: max.x.clamp(min.x, framebuffer_dimensions[0] as f32),
            y: max.y.clamp(min.y, framebuffer_dimensions[1] as f32),
        };
        Scissor {
            offset: [min.x.round() as u32, min.y.round() as u32],
            extent: [(max.x.round() - min.x) as u32, (max.y.round() - min.y) as u32],
        }
    }

    fn create_secondary_command_buffer_builder(
        &self,
    ) -> AutoCommandBufferBuilder<SecondaryAutoCommandBuffer> {
        AutoCommandBufferBuilder::secondary(
            self.allocators.command_buffer.clone(),
            self.gfx_queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
            CommandBufferInheritanceInfo {
                render_pass: Some(self.subpass.clone().into()),
                ..Default::default()
            },
        )
        .unwrap()
    }

    // Starts the rendering pipeline and returns [`RecordingCommandBuffer`] for drawing
    fn start(
        &mut self,
        final_image: Arc<ImageView>,
    ) -> (AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>, [u32; 2]) {
        // Get dimensions
        let img_dims = final_image.image().extent();
        // Create framebuffer (must be in same order as render pass description in `new`
        let framebuffer = Framebuffer::new(
            self.render_pass
                .as_ref()
                .expect(
                    "No renderpass on this renderer (created with subpass), use 'draw_subpass' \
                     instead",
                )
                .clone(),
            FramebufferCreateInfo { attachments: vec![final_image], ..Default::default() },
        )
        .unwrap();
        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            self.allocators.command_buffer.clone(),
            self.gfx_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        // Add clear values here for attachments and begin render pass
        command_buffer_builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![if !self.is_overlay { Some([0.0; 4].into()) } else { None }],
                    ..RenderPassBeginInfo::framebuffer(framebuffer)
                },
                SubpassBeginInfo {
                    contents: SubpassContents::SecondaryCommandBuffers,
                    ..SubpassBeginInfo::default()
                },
            )
            .unwrap();
        (command_buffer_builder, [img_dims[0], img_dims[1]])
    }

    /// Executes our draw commands on the final image and returns a `GpuFuture` to wait on
    pub fn draw_on_image<F>(
        &mut self,
        clipped_meshes: &[ClippedPrimitive],
        textures_delta: &TexturesDelta,
        scale_factor: f32,
        before_future: F,
        final_image: Arc<ImageView>,
    ) -> Box<dyn GpuFuture>
    where
        F: GpuFuture + 'static,
    {
        self.update_textures(&textures_delta.set);

        let (mut command_buffer_builder, framebuffer_dimensions) = self.start(final_image);
        let mut builder = self.create_secondary_command_buffer_builder();
        self.draw_egui(scale_factor, clipped_meshes, framebuffer_dimensions, &mut builder);
        // Execute draw commands
        let command_buffer = builder.build().unwrap();
        command_buffer_builder.execute_commands(command_buffer).unwrap();
        let done_future = self.finish(command_buffer_builder, Box::new(before_future));

        for &id in &textures_delta.free {
            self.unregister_image(id);
        }

        done_future
    }

    // Finishes the rendering pipeline
    fn finish(
        &self,
        mut command_buffer_builder: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        before_main_cb_future: Box<dyn GpuFuture>,
    ) -> Box<dyn GpuFuture> {
        // We end render pass
        command_buffer_builder.end_render_pass(Default::default()).unwrap();
        // Then execute our whole command buffer
        let command_buffer = command_buffer_builder.build().unwrap();
        let after_main_cb =
            before_main_cb_future.then_execute(self.gfx_queue.clone(), command_buffer).unwrap();
        // Return our future
        Box::new(after_main_cb)
    }

    pub fn draw_on_subpass_image(
        &mut self,
        clipped_meshes: &[ClippedPrimitive],
        textures_delta: &TexturesDelta,
        scale_factor: f32,
        framebuffer_dimensions: [u32; 2],
    ) -> Arc<SecondaryAutoCommandBuffer> {
        self.update_textures(&textures_delta.set);
        let mut builder = self.create_secondary_command_buffer_builder();
        self.draw_egui(scale_factor, clipped_meshes, framebuffer_dimensions, &mut builder);
        let buffer = builder.build().unwrap();
        for &id in &textures_delta.free {
            self.unregister_image(id);
        }
        buffer
    }
    /// Uploads all meshes in bulk. They will be available in the same order, packed.
    /// None if no vertices or no indices.
    fn upload_meshes(
        &mut self,
        clipped_meshes: &[ClippedPrimitive],
    ) -> Option<(VertexBuffer, IndexBuffer)> {
        use egui::epaint::Vertex;
        type Index = u32;
        const VERTEX_ALIGN: DeviceAlignment = DeviceAlignment::of::<Vertex>();
        const INDEX_ALIGN: DeviceAlignment = DeviceAlignment::of::<Index>();

        // Iterator over only the meshes, no user callbacks.
        let meshes = clipped_meshes.iter().filter_map(|mesh| match &mesh.primitive {
            Primitive::Mesh(m) => Some(m),
            _ => None,
        });

        // Calculate counts of each mesh, and total bytes for combined data
        let (total_vertices, total_size_bytes) = {
            let mut total_vertices = 0;
            let mut total_indices = 0;

            for mesh in meshes.clone() {
                total_vertices += mesh.vertices.len();
                total_indices += mesh.indices.len();
            }
            if total_indices == 0 || total_vertices == 0 {
                return None;
            }

            let total_size_bytes = total_vertices * std::mem::size_of::<Vertex>()
                + total_indices * std::mem::size_of::<Index>();
            (
                total_vertices,
                // Infallible! Checked above.
                NonZeroDeviceSize::new(u64::try_from(total_size_bytes).unwrap()).unwrap(),
            )
        };

        // Allocate a buffer which can hold both packed arrays:
        let layout = DeviceLayout::new(total_size_bytes, VERTEX_ALIGN.max(INDEX_ALIGN)).unwrap();
        let buffer = self.vertex_index_buffer_pool.allocate(layout).unwrap();

        // We must put the items with stricter align *first* in the packed buffer.
        // Correct at time of writing, but assert in case that changes.
        assert!(VERTEX_ALIGN >= INDEX_ALIGN);
        let (vertices, indices) = {
            let partition_bytes = total_vertices as u64 * std::mem::size_of::<Vertex>() as u64;
            (
                // Slice the start as vertices
                buffer.clone().slice(..partition_bytes).reinterpret::<[Vertex]>(),
                // Take the rest, reinterpret as indices.
                buffer.slice(partition_bytes..).reinterpret::<[Index]>(),
            )
        };

        // We have to upload in two mapping steps to avoid trivial but ugly unsafe.
        {
            let mut vertex_write = vertices.write().unwrap();
            vertex_write
                .iter_mut()
                .zip(meshes.clone().flat_map(|m| &m.vertices).copied())
                .for_each(|(into, from)| *into = from);
        }
        {
            let mut index_write = indices.write().unwrap();
            index_write
                .iter_mut()
                .zip(meshes.flat_map(|m| &m.indices).copied())
                .for_each(|(into, from)| *into = from);
        }

        Some((vertices, indices))
    }

    fn draw_egui(
        &mut self,
        scale_factor: f32,
        clipped_meshes: &[ClippedPrimitive],
        framebuffer_dimensions: [u32; 2],
        builder: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
    ) {
        let push_constants = vs::PushConstants {
            screen_size: [
                framebuffer_dimensions[0] as f32 / scale_factor,
                framebuffer_dimensions[1] as f32 / scale_factor,
            ],
            output_in_linear_colorspace: self.output_in_linear_colorspace.into(),
        };

        let mesh_buffers = self.upload_meshes(clipped_meshes);

        // Current position of renderbuffers, advances as meshes are consumed.
        let mut vertex_cursor = 0;
        let mut index_cursor = 0;
        // Some of our state is immutable and only changes
        // if a user callback thrashes it, rebind all when this is set:
        let mut needs_full_rebind = true;
        // Track resources that change from call-to-call.
        // egui already makes the optimization that draws with identical resources are merged into one,
        // so every mesh changes usually one or possibly both of these.
        let mut current_rect = None;
        let mut current_texture = None;

        for ClippedPrimitive { clip_rect, primitive } in clipped_meshes {
            match primitive {
                Primitive::Mesh(mesh) => {
                    // Nothing to draw if we don't have vertices & indices
                    if mesh.vertices.is_empty() || mesh.indices.is_empty() {
                        // Consume the mesh and skip it.
                        index_cursor += mesh.indices.len() as u32;
                        vertex_cursor += mesh.vertices.len() as u32;
                        continue;
                    }
                    // Reset overall state, if needed.
                    // Only happens on first mesh, and after a user callback which does unknowable
                    // things to the command buffer's state.
                    if needs_full_rebind {
                        needs_full_rebind = false;

                        // Bind combined meshes.
                        let Some((vertices, indices)) = mesh_buffers.clone() else {
                            // Only None if there are no mesh calls, but here we are in a mesh call!
                            unreachable!()
                        };

                        builder
                            .bind_pipeline_graphics(self.pipeline.clone())
                            .unwrap()
                            .bind_index_buffer(indices)
                            .unwrap()
                            .bind_vertex_buffers(0, [vertices])
                            .unwrap()
                            .set_viewport(
                                0,
                                [Viewport {
                                    offset: [0.0, 0.0],
                                    extent: [
                                        framebuffer_dimensions[0] as f32,
                                        framebuffer_dimensions[1] as f32,
                                    ],
                                    depth_range: 0.0..=1.0,
                                }]
                                .into_iter()
                                .collect(),
                            )
                            .unwrap()
                            .push_constants(self.pipeline.layout().clone(), 0, push_constants)
                            .unwrap();
                    }
                    // Find and bind image, if different.
                    if current_texture != Some(mesh.texture_id) {
                        if self.texture_desc_sets.get(&mesh.texture_id).is_none() {
                            eprintln!("This texture no longer exists {:?}", mesh.texture_id);
                            continue;
                        }
                        current_texture = Some(mesh.texture_id);

                        let desc_set = self.texture_desc_sets.get(&mesh.texture_id).unwrap();

                        builder
                            .bind_descriptor_sets(
                                PipelineBindPoint::Graphics,
                                self.pipeline.layout().clone(),
                                0,
                                desc_set.clone(),
                            )
                            .unwrap();
                    };
                    // Calculate and set scissor, if different
                    if current_rect != Some(*clip_rect) {
                        current_rect = Some(*clip_rect);
                        let new_scissor =
                            self.get_rect_scissor(scale_factor, framebuffer_dimensions, *clip_rect);

                        builder.set_scissor(0, [new_scissor].into_iter().collect()).unwrap();
                    }

                    // All set up to draw!
                    unsafe {
                        builder
                            .draw_indexed(
                                mesh.indices.len() as u32,
                                1,
                                index_cursor,
                                vertex_cursor as i32,
                                0,
                            )
                            .unwrap();
                    }

                    // Consume this mesh for next iteration
                    index_cursor += mesh.indices.len() as u32;
                    vertex_cursor += mesh.vertices.len() as u32;
                }
                Primitive::Callback(callback) => {
                    if callback.rect.is_positive() {
                        let Some(callback_fn) = callback.callback.downcast_ref::<CallbackFn>()
                        else {
                            println!(
                                "Warning: Unsupported render callback. Expected \
                                 egui_winit_vulkano::CallbackFn"
                            );
                            continue;
                        };

                        let rect_min_x = scale_factor * callback.rect.min.x;
                        let rect_min_y = scale_factor * callback.rect.min.y;
                        let rect_max_x = scale_factor * callback.rect.max.x;
                        let rect_max_y = scale_factor * callback.rect.max.y;

                        let rect_min_x = rect_min_x.round();
                        let rect_min_y = rect_min_y.round();
                        let rect_max_x = rect_max_x.round();
                        let rect_max_y = rect_max_y.round();

                        builder
                            .set_viewport(
                                0,
                                [Viewport {
                                    offset: [rect_min_x, rect_min_y],
                                    extent: [rect_max_x - rect_min_x, rect_max_y - rect_min_y],
                                    depth_range: 0.0..=1.0,
                                }]
                                .into_iter()
                                .collect(),
                            )
                            .unwrap()
                            .set_scissor(
                                0,
                                [self.get_rect_scissor(
                                    scale_factor,
                                    framebuffer_dimensions,
                                    *clip_rect,
                                )]
                                .into_iter()
                                .collect(),
                            )
                            .unwrap();

                        let info = egui::PaintCallbackInfo {
                            viewport: callback.rect,
                            clip_rect: *clip_rect,
                            pixels_per_point: scale_factor,
                            screen_size_px: framebuffer_dimensions,
                        };
                        (callback_fn.f)(info, &mut CallbackContext {
                            builder,
                            resources: self.render_resources(),
                        });

                        // The user could have done much here - rebind pipes, set views, bind things, etc.
                        // Mark all state as lost so that next mesh rebinds everything to a known state.
                        needs_full_rebind = true;
                        current_rect = None;
                        current_texture = None;
                    }
                }
            }
        }
    }

    pub fn render_resources(&self) -> RenderResources {
        RenderResources {
            queue: self.queue(),
            subpass: self.subpass.clone(),
            memory_allocator: self.allocators.memory.clone(),
            descriptor_set_allocator: &self.allocators.descriptor_set,
            command_buffer_allocator: &self.allocators.command_buffer,
        }
    }

    pub fn queue(&self) -> Arc<Queue> {
        self.gfx_queue.clone()
    }

    pub fn allocators(&self) -> &Allocators {
        &self.allocators
    }
}

/// A set of objects used to perform custom rendering in a `PaintCallback`.
///
/// It includes [`RenderResources`] for constructing a subpass pipeline and a secondary
/// command buffer for pushing render commands onto it.
///
/// # Example
///
/// See the `triangle` demo source for a detailed usage example.
pub struct CallbackContext<'a> {
    pub builder: &'a mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
    pub resources: RenderResources<'a>,
}

/// A set of resources used to construct the render pipeline. These can be reused
/// to create additional pipelines and buffers to be rendered in a `PaintCallback`.
///
/// # Example
///
/// See the `triangle` demo source for a detailed usage example.
#[derive(Clone)]
pub struct RenderResources<'a> {
    pub memory_allocator: Arc<StandardMemoryAllocator>,
    pub descriptor_set_allocator: &'a StandardDescriptorSetAllocator,
    pub command_buffer_allocator: &'a StandardCommandBufferAllocator,
    pub queue: Arc<Queue>,
    pub subpass: Subpass,
}

pub type CallbackFnDef = dyn Fn(PaintCallbackInfo, &mut CallbackContext) + Sync + Send;

/// A callback function that can be used to compose an [`epaint::PaintCallback`] for
/// custom rendering with [`vulkano`].
///
/// The callback is passed an [`egui::PaintCallbackInfo`] and a [`CallbackContext`] which
/// can be used to construct Vulkano graphics pipelines and buffers.
///
/// # Example
///
/// See the `triangle` demo source for a detailed usage example.
pub struct CallbackFn {
    pub(crate) f: Box<CallbackFnDef>,
}

impl CallbackFn {
    pub fn new<F: Fn(PaintCallbackInfo, &mut CallbackContext) + Sync + Send + 'static>(
        callback: F,
    ) -> Self {
        let f = Box::new(callback);
        CallbackFn { f }
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
    int output_in_linear_colorspace;
} push_constants;

void main() {
    gl_Position = vec4(
        2.0 * position.x / push_constants.screen_size.x - 1.0,
        2.0 * position.y / push_constants.screen_size.y - 1.0,
        0.0, 1.0
    );
    v_color = color;
    v_tex_coords = tex_coords;
}"
    }
}

// Similar to https://github.com/ArjunNair/egui_sdl2_gl/blob/main/src/painter.rs
mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
#version 450

layout(location = 0) in vec4 v_color;
layout(location = 1) in vec2 v_tex_coords;

layout(location = 0) out vec4 f_color;

layout(binding = 0, set = 0) uniform sampler2D font_texture;

layout(push_constant) uniform PushConstants {
    vec2 screen_size;
    int output_in_linear_colorspace;
} push_constants;

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
    vec4 texture_color = srgba_from_linear(texture(font_texture, v_tex_coords));
    vec4 color = v_color * texture_color;

    // If output_in_linear_colorspace is true, we are rendering into an sRGB image, for which we'll convert to linear color space.
    // **This will break blending** as it will be performed in linear color space instead of sRGB like egui expects.
    if (push_constants.output_in_linear_colorspace == 1) {
        color = linear_from_srgba(color);
    }
    f_color = color;
}"
    }
}
