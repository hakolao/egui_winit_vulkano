// Copyright (c) 2021 Okko Hakola
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;

use egui::{CtxRef, Visuals};
use egui_winit_vulkano::Gui;
use vulkano::{
    image::{view::ImageView, AttachmentImage},
    swapchain::PresentMode,
};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

use crate::{renderer::Renderer, time_info::TimeInfo};

mod frame_system;
mod renderer;
mod time_info;
mod triangle_draw_system;

/// Example struct to contain the state of the UI
pub struct GuiState {
    show_texture_window: bool,
    show_scene_window: bool,
    image_texture_id: egui::TextureId,
    scene_texture_ids: Vec<egui::TextureId>,
    scene_view_size: [u32; 2],
}

impl GuiState {
    pub fn new(
        gui: &mut Gui,
        scene_images: &Vec<Arc<ImageView<Arc<AttachmentImage>>>>,
        scene_view_size: [u32; 2],
    ) -> GuiState {
        // tree.png asset is from https://github.com/sotrh/learn-wgpu/tree/master/docs/beginner/tutorial5-textures
        let image_texture_id = gui.register_user_image(
            include_bytes!("./assets/tree.png"),
            vulkano::format::Format::R8G8B8A8Unorm,
        );
        let mut scene_texture_ids = vec![];
        for img in scene_images {
            scene_texture_ids.push(gui.register_user_image_view(img.clone()));
        }
        GuiState {
            show_texture_window: true,
            show_scene_window: true,
            image_texture_id,
            scene_texture_ids,
            scene_view_size,
        }
    }

    /// Defines the layout of our UI
    pub fn layout(
        &mut self,
        egui_context: CtxRef,
        window: &Window,
        last_image_num: usize,
        fps: f32,
    ) {
        egui_context.set_visuals(Visuals::dark());
        egui::SidePanel::left("Side Panel", 150.0).show(&egui_context, |ui| {
            ui.heading("Hello Tree");
            ui.separator();
            ui.checkbox(&mut self.show_texture_window, "Show Tree");
            ui.checkbox(&mut self.show_scene_window, "Show Scene");
        });
        let show_texture_window = &mut self.show_texture_window;
        let image_texture_id = self.image_texture_id;
        egui::Window::new("Mah Tree").resizable(true).scroll(true).open(show_texture_window).show(
            &egui_context,
            |ui| {
                ui.image(image_texture_id, [256.0, 256.0]);
            },
        );
        let show_scene_window = &mut self.show_scene_window;
        let scene_texture_id = self.scene_texture_ids[last_image_num];
        let scene_view_size = self.scene_view_size;
        egui::Window::new("Scene").resizable(true).scroll(true).open(show_scene_window).show(
            &egui_context,
            |ui| {
                ui.image(scene_texture_id, [scene_view_size[0] as f32, scene_view_size[1] as f32]);
            },
        );
        let size = window.inner_size();
        egui::Area::new("fps")
            .fixed_pos(egui::pos2(size.width as f32 - 0.05 * size.width as f32, 10.0))
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
    let window_size = [1280, 720];
    let scene_view_size = [256, 256];
    let mut renderer = Renderer::new(
        &event_loop,
        window_size,
        scene_view_size,
        PresentMode::Immediate,
        "Wholesome",
    );
    // After creating the renderer (window, gfx_queue) create out gui integration
    // It requires access to surface (Window, devices etc.) and Vulkano's gfx queue
    let mut gui = Gui::new(renderer.surface(), renderer.queue());
    // Renderer created AttachmentImages for our scene, let's access them
    let scene_images = renderer.scene_images();
    // Create gui state (pass anything your state requires)
    let mut gui_state = GuiState::new(&mut gui, scene_images, scene_view_size);
    event_loop.run(move |event, _, control_flow| {
        // Update Egui integration so the UI works!
        gui.update(&event);
        match event {
            Event::WindowEvent { event, window_id } if window_id == window_id => match event {
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
            },
            Event::RedrawRequested(window_id) if window_id == window_id => {
                // Set immediate UI in redraw here
                // It's a closure giving access to egui context inside which you can call anything.
                // Here we're calling the layout of our `gui_state`.
                gui.immediate_ui(|gui| {
                    let ctx = gui.context();
                    gui_state.layout(ctx, renderer.window(), renderer.last_image_num(), time.fps())
                });
                // Lastly we'll need to render our ui. You need to organize gui rendering to your needs
                // We'll render gui last on our swapchain images (see function below)
                renderer.render(&mut gui);
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
