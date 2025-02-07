// Copyright (c) 2021 Okko Hakola
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

#![allow(clippy::eq_op)]

use std::sync::Arc;

use egui::{epaint::Shadow, vec2, Align, Align2, Color32, CornerRadius, Frame, Margin, Window};
use egui_winit_vulkano::{Gui, GuiConfig};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        AutoCommandBufferBuilder, CommandBufferInheritanceInfo, CommandBufferUsage,
        RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
    },
    device::{Device, Queue},
    format::Format,
    image::{view::ImageView, Image, ImageCreateInfo, ImageType, ImageUsage, SampleCount},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
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
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    sync::GpuFuture,
};
use vulkano_util::{
    context::{VulkanoConfig, VulkanoContext},
    window::{VulkanoWindows, WindowDescriptor},
};
use winit::{
    application::ApplicationHandler, error::EventLoopError, event::WindowEvent,
    event_loop::EventLoop,
};

pub struct App {
    context: VulkanoContext,
    windows: VulkanoWindows,
    pipeline: Option<MSAAPipeline>,
    gui: Option<Gui>,
}

impl Default for App {
    fn default() -> Self {
        // Vulkano context
        let context = VulkanoContext::new(VulkanoConfig::default());

        // Vulkano windows
        let windows = VulkanoWindows::default();

        Self { context, windows, pipeline: None, gui: None }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        self.windows.create_window(event_loop, &self.context, &WindowDescriptor::default(), |ci| {
            ci.image_format = vulkano::format::Format::B8G8R8A8_UNORM;
            ci.image_usage = ImageUsage::TRANSFER_DST | ci.image_usage;
            ci.min_image_count = ci.min_image_count.max(2);
        });

        // Create out gui pipeline
        let pipeline = MSAAPipeline::new(
            self.context.graphics_queue().clone(),
            self.windows.get_primary_renderer_mut().unwrap().swapchain_format(),
            self.context.memory_allocator(),
            SampleCount::Sample4,
        );

        // Create gui subpass
        self.gui = Some(Gui::new_with_subpass(
            event_loop,
            self.windows.get_primary_renderer_mut().unwrap().surface(),
            self.windows.get_primary_renderer_mut().unwrap().graphics_queue(),
            pipeline.gui_pass(),
            self.windows.get_primary_renderer_mut().unwrap().swapchain_format(),
            GuiConfig {
                // Must match your pipeline's sample count
                samples: SampleCount::Sample4,
                ..Default::default()
            },
        ));

        self.pipeline = Some(pipeline);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let renderer = self.windows.get_renderer_mut(window_id).unwrap();

        let gui = self.gui.as_mut().unwrap();

        // Update Egui integration so the UI works!
        let _pass_events_to_game = !gui.update(&event);
        match event {
            WindowEvent::Resized(_) => {
                renderer.resize();
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                renderer.resize();
            }
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                // Set immediate UI in redraw here
                gui.immediate_ui(|gui| {
                    let ctx = gui.context();
                    Window::new("Transparent Window")
                        .anchor(Align2([Align::RIGHT, Align::TOP]), vec2(-545.0, 500.0))
                        .resizable(false)
                        .default_width(300.0)
                        .frame(
                            Frame::NONE
                                .fill(Color32::from_white_alpha(125))
                                .shadow(Shadow {
                                    spread: 8,
                                    blur: 10,
                                    color: Color32::from_black_alpha(125),
                                    ..Default::default()
                                })
                                .corner_radius(CornerRadius::same(5))
                                .inner_margin(Margin::same(10)),
                        )
                        .show(&ctx, |ui| {
                            ui.colored_label(Color32::BLACK, "Content :)");
                        });
                });
                // Render
                // Acquire swapchain future
                match renderer.acquire(Some(std::time::Duration::from_millis(10)), |_| {}) {
                    Ok(future) => {
                        // Render
                        let after_future = self.pipeline.as_mut().unwrap().render(
                            future,
                            renderer.swapchain_image_view(),
                            gui,
                        );
                        // Present swapchain
                        renderer.present(after_future, true);
                    }
                    Err(vulkano::VulkanError::OutOfDate) => {
                        renderer.resize();
                    }
                    Err(e) => panic!("Failed to acquire swapchain future: {}", e),
                };
            }
            _ => (),
        }
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        let renderer = self.windows.get_primary_renderer().unwrap();
        renderer.window().request_redraw();
    }
}

pub fn main() -> Result<(), EventLoopError> {
    // Winit event loop
    let event_loop = EventLoop::new().unwrap();

    let mut app = App::default();

    event_loop.run_app(&mut app)
}

struct MSAAPipeline {
    allocator: Arc<StandardMemoryAllocator>,
    queue: Arc<Queue>,
    render_pass: Arc<RenderPass>,
    pipeline: Arc<GraphicsPipeline>,
    subpass: Subpass,
    intermediary: Arc<ImageView>,
    vertex_buffer: Subbuffer<[MyVertex]>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
}

impl MSAAPipeline {
    pub fn new(
        queue: Arc<Queue>,
        image_format: vulkano::format::Format,
        allocator: &Arc<StandardMemoryAllocator>,
        sample_count: SampleCount,
    ) -> Self {
        let render_pass =
            Self::create_render_pass(queue.device().clone(), image_format, sample_count);
        let (pipeline, subpass) =
            Self::create_pipeline(queue.device().clone(), render_pass.clone());

        let vertex_buffer = Buffer::from_iter(
            allocator.clone(),
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

        // Create an allocator for command-buffer data
        let command_buffer_allocator = StandardCommandBufferAllocator::new(
            queue.device().clone(),
            StandardCommandBufferAllocatorCreateInfo {
                secondary_buffer_count: 32,
                ..Default::default()
            },
        )
        .into();

        let intermediary = ImageView::new_default(
            Image::new(
                allocator.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: image_format,
                    extent: [1, 1, 1],
                    usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
                    samples: sample_count,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap(),
        )
        .unwrap();

        Self {
            allocator: allocator.clone(),
            queue,
            render_pass,
            pipeline,
            subpass,
            intermediary,
            vertex_buffer,
            command_buffer_allocator,
        }
    }

    fn create_render_pass(
        device: Arc<Device>,
        format: Format,
        samples: SampleCount,
    ) -> Arc<RenderPass> {
        vulkano::single_pass_renderpass!(
            device,
            attachments: {
                // The first framebuffer attachment is the intermediary image.
                intermediary: {
                    format: format,
                    samples: samples,
                    load_op: Clear,
                    store_op: DontCare,
                },
                // The second framebuffer attachment is the final image.
                color: {
                    format: format,
                    samples: 1,
                    load_op: DontCare,
                    store_op: Store,
                }
            },
            pass: {
                color: [intermediary],
                color_resolve: [color],
                depth_stencil: {},
            }
        )
        .unwrap()
    }

    fn gui_pass(&self) -> Subpass {
        self.subpass.clone()
    }

    fn create_pipeline(
        device: Arc<Device>,
        render_pass: Arc<RenderPass>,
    ) -> (Arc<GraphicsPipeline>, Subpass) {
        let vs = vs::load(device.clone())
            .expect("failed to create shader module")
            .entry_point("main")
            .unwrap();
        let fs = fs::load(device.clone())
            .expect("failed to create shader module")
            .entry_point("main")
            .unwrap();

        let vertex_input_state = MyVertex::per_vertex().definition(&vs).unwrap();

        let stages =
            [PipelineShaderStageCreateInfo::new(vs), PipelineShaderStageCreateInfo::new(fs)];

        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();

        let subpass = Subpass::from(render_pass, 0).unwrap();
        (
            GraphicsPipeline::new(device.clone(), None, GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState::default()),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState {
                    rasterization_samples: subpass.num_samples().unwrap(),
                    ..MultisampleState::default()
                }),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),
                dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                subpass: Some(subpass.clone().into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            })
            .unwrap(),
            subpass,
        )
    }

    pub fn render(
        &mut self,
        before_future: Box<dyn GpuFuture>,
        image: Arc<ImageView>,
        gui: &mut Gui,
    ) -> Box<dyn GpuFuture> {
        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let dimensions = image.image().extent();
        // Resize intermediary image
        if dimensions != self.intermediary.image().extent() {
            self.intermediary = ImageView::new_default(
                Image::new(
                    self.allocator.clone(),
                    ImageCreateInfo {
                        image_type: ImageType::Dim2d,
                        format: image.image().format(),
                        extent: image.image().extent(),
                        // transient_multisampled
                        usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED,
                        samples: self.subpass.num_samples().unwrap(),
                        ..Default::default()
                    },
                    AllocationCreateInfo::default(),
                )
                .unwrap(),
            )
            .unwrap();
        }

        let framebuffer = Framebuffer::new(self.render_pass.clone(), FramebufferCreateInfo {
            attachments: vec![self.intermediary.clone(), image],
            ..Default::default()
        })
        .unwrap();

        // Begin render pipeline commands
        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.0, 0.0, 0.0, 1.0].into()), None],
                    ..RenderPassBeginInfo::framebuffer(framebuffer)
                },
                SubpassBeginInfo {
                    contents: SubpassContents::SecondaryCommandBuffers,
                    ..Default::default()
                },
            )
            .unwrap();

        // Render first draw pass
        let mut secondary_builder = AutoCommandBufferBuilder::secondary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
            CommandBufferInheritanceInfo {
                render_pass: Some(self.subpass.clone().into()),
                ..Default::default()
            },
        )
        .unwrap();
        secondary_builder
            .bind_pipeline_graphics(self.pipeline.clone())
            .unwrap()
            .set_viewport(
                0,
                [Viewport {
                    offset: [0.0, 0.0],
                    extent: [dimensions[0] as f32, dimensions[1] as f32],
                    depth_range: 0.0..=1.0,
                }]
                .into_iter()
                .collect(),
            )
            .unwrap()
            .bind_vertex_buffers(0, self.vertex_buffer.clone())
            .unwrap();
        unsafe {
            secondary_builder.draw(self.vertex_buffer.len() as u32, 1, 0, 0).unwrap();
        }
        let cb = secondary_builder.build().unwrap();
        builder.execute_commands(cb).unwrap();

        // Draw gui on subpass
        let cb = gui.draw_on_subpass_image([dimensions[0], dimensions[1]]);
        builder.execute_commands(cb).unwrap();

        // Last end render pass
        builder.end_render_pass(Default::default()).unwrap();
        let command_buffer = builder.build().unwrap();
        let after_future = before_future.then_execute(self.queue.clone(), command_buffer).unwrap();

        after_future.boxed()
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
