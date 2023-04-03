// Copyright (c) 2021 Okko Hakola
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::{convert::TryInto, sync::Arc};

use egui::{mutex::Mutex, vec2, PaintCallback, PaintCallbackInfo, Rgba, Sense};
use egui_winit_vulkano::{CallbackContext, CallbackFn, Gui, GuiConfig, RenderResources};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{AllocationCreateInfo, MemoryUsage},
    pipeline::{
        graphics::{
            depth_stencil::DepthStencilState, input_assembly::InputAssemblyState,
            vertex_input::Vertex, viewport::ViewportState,
        },
        GraphicsPipeline,
    },
};
use vulkano_util::{
    context::{VulkanoConfig, VulkanoContext},
    window::{VulkanoWindows, WindowDescriptor},
};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

pub fn main() {
    // Winit event loop
    let event_loop = EventLoop::new();
    // Vulkano context
    let context = VulkanoContext::new(VulkanoConfig::default());
    // Vulkano windows (create one)
    let mut windows = VulkanoWindows::default();
    windows.create_window(
        &event_loop,
        &context,
        &WindowDescriptor { width: 400.0, height: 400.0, ..Default::default() },
        |ci| ci.image_format = Some(vulkano::format::Format::B8G8R8A8_SRGB),
    );
    // Create gui as main render pass (no overlay means it clears the image each frame)
    let (mut gui, scene) = {
        let renderer = windows.get_primary_renderer_mut().unwrap();

        let gui = Gui::new(&event_loop, renderer.surface(), renderer.graphics_queue(), GuiConfig {
            preferred_format: Some(vulkano::format::Format::B8G8R8A8_SRGB),
            ..Default::default()
        });

        let scene = Arc::new(Mutex::new(Scene::new(gui.render_resources())));

        (gui, scene)
    };

    // Create gui state (pass anything your state requires)
    event_loop.run(move |event, _, control_flow| {
        let renderer = windows.get_primary_renderer_mut().unwrap();
        match event {
            Event::WindowEvent { event, window_id } if window_id == renderer.window().id() => {
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
                        *control_flow = ControlFlow::Exit;
                    }
                    _ => (),
                }
            }
            Event::RedrawRequested(window_id) if window_id == renderer.window().id() => {
                let scene = scene.clone();
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
                let before_future = renderer.acquire().unwrap();
                // Render gui
                let after_future =
                    gui.draw_on_image(before_future, renderer.swapchain_image_view());
                // Present swapchain
                renderer.present(after_future, true);
            }
            Event::MainEventsCleared => {
                renderer.window().request_redraw();
            }
            _ => (),
        }
    });
}

struct Scene {
    pipeline: Arc<GraphicsPipeline>,
    vertex_buffer: Subbuffer<[MyVertex]>,
}
impl Scene {
    pub fn new(resources: RenderResources) -> Self {
        // Create the vertex buffer for the triangle
        let vertex_buffer = Buffer::from_iter(
            &resources.memory_allocator,
            BufferCreateInfo { usage: BufferUsage::VERTEX_BUFFER, ..Default::default() },
            AllocationCreateInfo { usage: MemoryUsage::Upload, ..Default::default() },
            [
                MyVertex { position: [-0.5, -0.25], color: [1.0, 0.0, 0.0, 1.0] },
                MyVertex { position: [0.0, 0.5], color: [0.0, 1.0, 0.0, 1.0] },
                MyVertex { position: [0.25, -0.1], color: [0.0, 0.0, 1.0, 1.0] },
            ],
        )
        .unwrap();

        // Create the graphics pipeline
        let vs =
            vs::load(resources.queue.device().clone()).expect("failed to create shader module");
        let fs =
            fs::load(resources.queue.device().clone()).expect("failed to create shader module");

        let pipeline = GraphicsPipeline::start()
            .vertex_input_state(MyVertex::per_vertex())
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            .input_assembly_state(InputAssemblyState::new())
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .depth_stencil_state(DepthStencilState::simple_depth_test())
            .render_pass(resources.subpass)
            .build(resources.queue.device().clone())
            .unwrap();

        // Create scene object
        Self { pipeline, vertex_buffer }
    }

    pub fn render(&mut self, _info: PaintCallbackInfo, context: &mut CallbackContext) {
        // Add the scene's rendering commands to the command buffer
        context
            .builder
            .bind_pipeline_graphics(self.pipeline.clone())
            .bind_vertex_buffers(0, self.vertex_buffer.clone())
            .draw(self.vertex_buffer.len() as u32, 1, 0, 0)
            .unwrap();
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
