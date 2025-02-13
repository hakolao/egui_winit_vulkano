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

use egui::{load::SizedTexture, Context, ImageSource, Visuals};
use egui_winit_vulkano::{Gui, GuiConfig};
use vulkano::{
    command_buffer::allocator::{
        StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
    },
    format::Format,
    image::{view::ImageView, Image, ImageCreateInfo, ImageType, ImageUsage},
    memory::allocator::AllocationCreateInfo,
};
use vulkano_util::{
    context::{VulkanoConfig, VulkanoContext},
    renderer::DEFAULT_IMAGE_FORMAT,
    window::{VulkanoWindows, WindowDescriptor},
};
use winit::{application::ApplicationHandler, event::WindowEvent, event_loop::EventLoop};

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
    pub fn new(gui: &mut Gui, scene_image: Arc<ImageView>, scene_view_size: [u32; 2]) -> GuiState {
        // tree.png asset is from https://github.com/sotrh/learn-wgpu/tree/master/docs/beginner/tutorial5-textures
        let image_texture_id1 = gui.register_user_image(
            include_bytes!("./assets/tree.png"),
            Format::R8G8B8A8_SRGB,
            Default::default(),
        );
        let image_texture_id2 = gui.register_user_image(
            include_bytes!("./assets/doge2.png"),
            Format::R8G8B8A8_SRGB,
            Default::default(),
        );

        GuiState {
            show_texture_window1: true,
            show_texture_window2: true,
            show_scene_window: true,
            image_texture_id1,
            image_texture_id2,
            scene_texture_id: gui.register_user_image_view(scene_image, Default::default()),
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
                ui.image(ImageSource::Texture(SizedTexture::new(*image_texture_id1, [
                    256.0, 256.0,
                ])));
            });
        egui::Window::new("Mah Doge")
            .resizable(true)
            .vscroll(true)
            .open(show_texture_window2)
            .show(&egui_context, |ui| {
                ui.image(ImageSource::Texture(SizedTexture::new(*image_texture_id2, [
                    300.0, 200.0,
                ])));
            });
        egui::Window::new("Scene").resizable(true).vscroll(true).open(show_scene_window).show(
            &egui_context,
            |ui| {
                ui.image(ImageSource::Texture(SizedTexture::new(*scene_texture_id, [
                    scene_view_size[0] as f32,
                    scene_view_size[1] as f32,
                ])));
            },
        );
        egui::Area::new("fps".into())
            .fixed_pos(egui::pos2(window_size[0] - 0.05 * window_size[0], 10.0))
            .show(&egui_context, |ui| {
                ui.label(format!("{fps:.2}"));
            });
    }
}

pub struct App {
    context: VulkanoContext,
    windows: VulkanoWindows,
    scene_view_size: [u32; 2],
    scene_image: Arc<ImageView>,
    time: TimeInfo,
    scene_render_pipeline: RenderPipeline,
    gui: Option<Gui>,
    gui_state: Option<GuiState>,
}

impl Default for App {
    fn default() -> Self {
        // Vulkano context
        let context = VulkanoContext::new(VulkanoConfig::default());

        // Vulkano windows
        let windows = VulkanoWindows::default();

        // Create renderer for our scene & ui
        let scene_view_size = [256, 256];
        // Create a simple image to which we'll draw the triangle scene
        let scene_image = ImageView::new_default(
            Image::new(
                context.memory_allocator().clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: DEFAULT_IMAGE_FORMAT,
                    extent: [scene_view_size[0], scene_view_size[1], 1],
                    array_layers: 1,
                    usage: ImageUsage::SAMPLED | ImageUsage::COLOR_ATTACHMENT,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap(),
        )
        .unwrap();

        let time = TimeInfo::new();

        // Create our render pipeline
        let scene_render_pipeline = RenderPipeline::new(
            context.graphics_queue().clone(),
            DEFAULT_IMAGE_FORMAT,
            &renderer::Allocators {
                command_buffers: Arc::new(StandardCommandBufferAllocator::new(
                    context.device().clone(),
                    StandardCommandBufferAllocatorCreateInfo {
                        secondary_buffer_count: 32,
                        ..Default::default()
                    },
                )),
                memory: context.memory_allocator().clone(),
            },
        );

        Self {
            context,
            windows,
            scene_view_size,
            scene_image,
            time,
            scene_render_pipeline,
            gui: None,
            gui_state: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        self.windows.create_window(event_loop, &self.context, &WindowDescriptor::default(), |ci| {
            ci.image_format = Format::B8G8R8A8_UNORM;
            ci.min_image_count = ci.min_image_count.max(2);
        });
        // Create gui as main render pass (no overlay means it clears the image each frame)
        let mut gui = {
            let renderer = self.windows.get_primary_renderer_mut().unwrap();
            Gui::new(
                event_loop,
                renderer.surface(),
                renderer.graphics_queue(),
                renderer.swapchain_format(),
                GuiConfig::default(),
            )
        };

        // Create gui state (pass anything your state requires)
        self.gui_state =
            Some(GuiState::new(&mut gui, self.scene_image.clone(), self.scene_view_size));

        self.gui = Some(gui);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let renderer = self.windows.get_renderer_mut(window_id).unwrap();

        let gui = self.gui.as_mut().unwrap();

        if window_id == renderer.window().id() {
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
                    // Set immediate UI in redraw here
                    // It's a closure giving access to egui context inside which you can call anything.
                    // Here we're calling the layout of our `gui_state`.
                    gui.immediate_ui(|gui| {
                        let ctx = gui.context();
                        self.gui_state.as_mut().unwrap().layout(
                            ctx,
                            renderer.window_size(),
                            self.time.fps(),
                        )
                    });
                    // Render UI
                    // Acquire swapchain future
                    match renderer.acquire(Some(std::time::Duration::from_millis(10)), |_| {}) {
                        Ok(future) => {
                            // Draw scene
                            let after_scene_draw =
                                self.scene_render_pipeline.render(future, self.scene_image.clone());
                            // Render gui
                            let after_future = gui
                                .draw_on_image(after_scene_draw, renderer.swapchain_image_view());
                            // Present swapchain
                            renderer.present(after_future, true);
                        }
                        Err(vulkano::VulkanError::OutOfDate) => {
                            renderer.resize();
                        }
                        Err(e) => panic!("Failed to acquire swapchain future: {}", e),
                    };

                    // Update fps & dt
                    self.time.update();
                }
                _ => (),
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        let renderer = self.windows.get_primary_renderer_mut().unwrap();
        renderer.window().request_redraw();
    }
}

pub fn main() -> Result<(), winit::error::EventLoopError> {
    // Winit event loop & our time tracking initialization
    let event_loop = EventLoop::new().unwrap();

    let mut app = App::default();

    // Event loop run
    event_loop.run_app(&mut app)
}
