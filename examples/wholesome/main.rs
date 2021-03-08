use std::sync::Arc;

use egui::{CtxRef, Visuals};
use egui_winit_vulkano::Gui;
use vulkano::{image::AttachmentImage, swapchain::PresentMode};
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
        scene_images: &Vec<Arc<AttachmentImage>>,
        scene_view_size: [u32; 2],
    ) -> GuiState {
        let image_texture_id = gui.register_user_image(include_bytes!("./assets/tree.png"));
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
    let event_loop = EventLoop::new();
    let mut time = TimeInfo::new();
    // Create renderer for our scene & ui
    let scene_view_size = [256, 256];
    let mut renderer = Renderer::new(
        &event_loop,
        1280,
        720,
        scene_view_size,
        PresentMode::Immediate,
        "Basic Example",
    );
    // After creating the renderer (window, gfx_queue) create out gui integration
    let mut gui = Gui::new(renderer.surface(), renderer.queue(), renderer.deferred_subpass());
    let scene_images = renderer.scene_images();
    // Create gui state (this should occur after renderer so we have access to gfx queue etc.)
    let mut gui_state = GuiState::new(&mut gui, scene_images, scene_view_size);
    event_loop.run(move |event, _, control_flow| {
        // Update Egui integration
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
                    return;
                }
                _ => (),
            },
            Event::RedrawRequested(window_id) if window_id == window_id => {
                // Set immediate UI in redraw here
                // It's a closure giving access to egui context inside which you can call anything.
                // Here we're calling the layout of our `gui_state`.
                gui.immediate_ui(|ctx| {
                    gui_state.layout(ctx, renderer.window(), renderer.last_image_num(), time.fps())
                });
                // Lastly we'll render our ui
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
