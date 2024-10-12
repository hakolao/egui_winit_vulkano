// Copyright (c) 2021 Okko Hakola
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;

use egui::{mutex::Mutex, vec2, PaintCallback, PaintCallbackInfo, Rgba, Sense};
use egui_winit_vulkano::{CallbackContext, CallbackFn, Gui, GuiConfig, RenderResources};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::ViewportState,
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    },
};
use vulkano_util::{
    context::{VulkanoConfig, VulkanoContext},
    window::{VulkanoWindows, WindowDescriptor},
};
use winit::{application::ApplicationHandler, event::WindowEvent, event_loop::EventLoop};

pub struct App {
    context: VulkanoContext,
    windows: VulkanoWindows,
    gui: Option<Gui>,
    scene: Option<Arc<Mutex<Scene>>>,
}

impl Default for App {
    fn default() -> Self {
        // Vulkano context
        let context = VulkanoContext::new(VulkanoConfig::default());

        // Vulkano windows
        let windows = VulkanoWindows::default();

        Self { context, windows, gui: None, scene: None }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        self.windows.create_window(
            event_loop,
            &self.context,
            &WindowDescriptor { width: 400.0, height: 400.0, ..Default::default() },
            |ci| {
                ci.image_format = vulkano::format::Format::B8G8R8A8_UNORM;
                ci.min_image_count = ci.min_image_count.max(2);
            },
        );

        // Create gui as main render pass (no overlay means it clears the image each frame)
        let renderer = self.windows.get_primary_renderer_mut().unwrap();

        let gui = Gui::new(
            event_loop,
            renderer.surface(),
            renderer.graphics_queue(),
            renderer.swapchain_format(),
            GuiConfig::default(),
        );

        let scene = Arc::new(Mutex::new(Scene::new(gui.render_resources())));

        self.gui = Some(gui);
        self.scene = Some(scene);
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
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                let scene = self.scene.clone().unwrap();
                // Set immediate UI in redraw here
                gui.immediate_ui(|gui| {
                    let ctx = gui.context();
                    egui::CentralPanel::default().show(&ctx, |ui| {
                        // Create a frame to render our triangle image in
                        egui::Frame::canvas(ui.style()).fill(Rgba::BLACK.into()).show(ui, |ui| {
                            // Allocate all the space in the frame for the image
                            let (rect, _) = ui.allocate_exact_size(
                                vec2(ui.available_width(), ui.available_height()),
                                Sense::click(),
                            );

                            // Render the scene in the allocated space
                            let paint_callback = PaintCallback {
                                rect,
                                callback: Arc::new(CallbackFn::new(move |info, context| {
                                    let mut scene = scene.lock();
                                    scene.render(info, context);
                                })),
                            };

                            ui.painter().add(paint_callback);
                        });
                    });
                });
                // Render UI
                // Acquire swapchain future
                match renderer.acquire(Some(std::time::Duration::from_millis(10)), |_| {}) {
                    Ok(future) => {
                        // Render gui
                        let after_future =
                            gui.draw_on_image(future, renderer.swapchain_image_view());
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

pub fn main() -> Result<(), winit::error::EventLoopError> {
    // Winit event loop
    let event_loop = EventLoop::new().unwrap();

    let mut app = App::default();

    event_loop.run_app(&mut app)
}

struct Scene {
    pipeline: Arc<GraphicsPipeline>,
    vertex_buffer: Subbuffer<[MyVertex]>,
}
impl Scene {
    pub fn new(resources: RenderResources) -> Self {
        // Create the vertex buffer for the triangle
        let vertex_buffer = Buffer::from_iter(
            resources.memory_allocator.clone(),
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

        // Create the graphics pipeline
        let vs = vs::load(resources.queue.device().clone())
            .expect("failed to create shader module")
            .entry_point("main")
            .unwrap();
        let fs = fs::load(resources.queue.device().clone())
            .expect("failed to create shader module")
            .entry_point("main")
            .unwrap();

        let vertex_input_state = MyVertex::per_vertex().definition(&vs).unwrap();

        let stages =
            [PipelineShaderStageCreateInfo::new(vs), PipelineShaderStageCreateInfo::new(fs)];

        let layout = PipelineLayout::new(
            resources.queue.device().clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(resources.queue.device().clone())
                .unwrap(),
        )
        .unwrap();

        let pipeline = GraphicsPipeline::new(
            resources.queue.device().clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState::default()),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    resources.subpass.num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),
                dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                subpass: Some(resources.subpass.clone().into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .unwrap();

        // Create scene object
        Self { pipeline, vertex_buffer }
    }

    pub fn render(&mut self, _info: PaintCallbackInfo, context: &mut CallbackContext) {
        // Add the scene's rendering commands to the command buffer
        context
            .builder
            .bind_pipeline_graphics(self.pipeline.clone())
            .unwrap()
            .bind_vertex_buffers(0, self.vertex_buffer.clone())
            .unwrap();
        unsafe {
            context.builder.draw(self.vertex_buffer.len() as u32, 1, 0, 0).unwrap();
        }
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
