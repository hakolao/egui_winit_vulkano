// Copyright (c) 2021 Okko Hakola
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;

use egui::{Context, Visuals};
use egui_winit_vulkano::Gui;
use vulkano::{
    command_buffer::allocator::StandardCommandBufferAllocator,
    format::Format,
    image::{ImageUsage, StorageImage},
};
use vulkano_util::{
    context::{VulkanoConfig, VulkanoContext},
    renderer::{DeviceImageView, DEFAULT_IMAGE_FORMAT},
    window::{VulkanoWindows, WindowDescriptor},
};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

use crate::{renderer::RenderPipeline, time_info::TimeInfo};

mod frame_system;
mod renderer;
mod time_info;
mod triangle_draw_system;

/// Example struct to contain the state of the UI
pub struct GuiState {
    show_texture_window1: bool,
    show_texture_window2: bool,
    show_scene_window: bool,
    image_texture_id1: egui::TextureId,
    image_texture_id2: egui::TextureId,
    scene_texture_id: egui::TextureId,
    scene_view_size: [u32; 2],
}

impl GuiState {
    pub fn new(gui: &mut Gui, scene_image: DeviceImageView, scene_view_size: [u32; 2]) -> GuiState {
        // tree.png asset is from https://github.com/sotrh/learn-wgpu/tree/master/docs/beginner/tutorial5-textures
        let image_texture_id1 =
            gui.register_user_image(include_bytes!("./assets/tree.png"), Format::R8G8B8A8_SRGB);
        let image_texture_id2 =
            gui.register_user_image(include_bytes!("./assets/doge2.png"), Format::R8G8B8A8_SRGB);

        GuiState {
            show_texture_window1: true,
            show_texture_window2: true,
            show_scene_window: true,
            image_texture_id1,
            image_texture_id2,
            scene_texture_id: gui.register_user_image_view(scene_image.clone()),
            scene_view_size,
        }
    }

    /// Defines the layout of our UI
    pub fn layout(&mut self, egui_context: Context, window_size: [f32; 2], fps: f32) {
        let GuiState {
            show_texture_window1,
            show_texture_window2,
            show_scene_window,
            image_texture_id1,
            image_texture_id2,
            scene_view_size,
            scene_texture_id,
            ..
        } = self;
        egui_context.set_visuals(Visuals::dark());
        egui::SidePanel::left("Side Panel").default_width(150.0).show(&egui_context, |ui| {
            ui.heading("Hello Tree");
            ui.separator();
            ui.checkbox(show_texture_window1, "Show Tree");
            ui.checkbox(show_texture_window2, "Show Doge");
            ui.checkbox(show_scene_window, "Show Scene");
        });

        egui::Window::new("Mah Tree")
            .resizable(true)
            .vscroll(true)
            .open(show_texture_window1)
            .show(&egui_context, |ui| {
                ui.image(*image_texture_id1, [256.0, 256.0]);
            });
        egui::Window::new("Mah Doge")
            .resizable(true)
            .vscroll(true)
            .open(show_texture_window2)
            .show(&egui_context, |ui| {
                ui.image(*image_texture_id2, [300.0, 200.0]);
            });
        egui::Window::new("Scene").resizable(true).vscroll(true).open(show_scene_window).show(
            &egui_context,
            |ui| {
                ui.image(*scene_texture_id, [scene_view_size[0] as f32, scene_view_size[1] as f32]);
            },
        );
        egui::Area::new("fps")
            .fixed_pos(egui::pos2(window_size[0] - 0.05 * window_size[0], 10.0))
            .show(&egui_context, |ui| {
                ui.label(format!("{:.2}", fps));
            });
    }
}

pub fn main() {
    // Winit event loop & our time tracking initialization
    let event_loop = EventLoop::new();
    let mut time = TimeInfo::new();
    // Create renderer for our scene & ui
    let scene_view_size = [256, 256];
    // Vulkano context
    let context = VulkanoContext::new(VulkanoConfig::default());
    // Vulkano windows (create one)
    let mut windows = VulkanoWindows::default();
    windows.create_window(&event_loop, &context, &WindowDescriptor::default(), |ci| {
        ci.image_format = Some(vulkano::format::Format::B8G8R8A8_SRGB)
    });
    // Create gui as main render pass (no overlay means it clears the image each frame)
    let mut gui = {
        let renderer = windows.get_primary_renderer_mut().unwrap();
        Gui::new(&event_loop, renderer.surface(), None, renderer.graphics_queue(), false)
    };
    // Create a simple image to which we'll draw the triangle scene
    let scene_image = StorageImage::general_purpose_image_view(
        context.memory_allocator(),
        context.graphics_queue().clone(),
        scene_view_size,
        DEFAULT_IMAGE_FORMAT,
        ImageUsage { sampled: true, color_attachment: true, ..ImageUsage::empty() },
    )
    .unwrap();
    // Create our render pipeline
    let mut scene_render_pipeline = RenderPipeline::new(
        context.graphics_queue().clone(),
        DEFAULT_IMAGE_FORMAT,
        &renderer::Allocators {
            command_buffers: Arc::new(StandardCommandBufferAllocator::new(
                context.device().clone(),
                Default::default(),
            )),
            memory: context.memory_allocator().clone(),
        },
    );
    // Create gui state (pass anything your state requires)
    let mut gui_state = GuiState::new(&mut gui, scene_image.clone(), scene_view_size);
    // Event loop run
    event_loop.run(move |event, _, control_flow| {
        let renderer = windows.get_primary_renderer_mut().unwrap();
        // Update Egui integration so the UI works!
        match event {
            Event::WindowEvent { event, window_id } if window_id == renderer.window().id() => {
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
            Event::RedrawRequested(window_id) if window_id == window_id => {
                // Set immediate UI in redraw here
                // It's a closure giving access to egui context inside which you can call anything.
                // Here we're calling the layout of our `gui_state`.
                gui.immediate_ui(|gui| {
                    let ctx = gui.context();
                    gui_state.layout(ctx, renderer.window_size(), time.fps())
                });
                // Render UI
                // Acquire swapchain future
                let before_future = renderer.acquire().unwrap();
                // Draw scene
                let after_scene_draw =
                    scene_render_pipeline.render(before_future, scene_image.clone());
                // Render gui
                let after_future =
                    gui.draw_on_image(after_scene_draw, renderer.swapchain_image_view());
                // Present swapchain
                renderer.present(after_future, true);

                // Update fps & dt
                time.update();
            }
            Event::MainEventsCleared => {
                renderer.window().request_redraw();
            }
            _ => (),
        }
    });
}
