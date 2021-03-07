use std::sync::Arc;

use egui::{CtxRef, Visuals};
use egui_winit_vulkan::EguiIntegration;
use vulkano::{device::Queue, swapchain::PresentMode};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

use crate::renderer::VulkanoWinitRenderer;

mod frame_system;
mod renderer;

/// Contains the state of our UI
pub struct AppGuiState {
    show_texture_window: bool,
    image_texture_id: egui::TextureId,
}

impl AppGuiState {
    pub fn new(gui: &mut EguiIntegration) -> AppGuiState {
        let image_texture_id = gui.register_user_image(include_bytes!("./assets/tree.png"));
        AppGuiState { show_texture_window: false, image_texture_id }
    }

    /// Defines the layout of our UI
    pub fn layout(&mut self, egui_context: CtxRef) {
        egui_context.set_visuals(Visuals::dark());
        egui::SidePanel::left("Side Panel", 150.0).show(&egui_context, |ui| {
            ui.heading("Debug");
            ui.separator();
            ui.checkbox(&mut self.show_texture_window, "Show Texture");
        });
        let show_texture_window = &mut self.show_texture_window;
        let image_texture_id = self.image_texture_id;
        egui::Window::new("Mah Tree").resizable(true).scroll(true).open(show_texture_window).show(
            &egui_context,
            |ui| {
                ui.image(image_texture_id, [256.0, 256.0]);
            },
        );
    }
}

pub fn main() {
    let event_loop = EventLoop::new();
    // Create gui integration shell
    let mut gui = EguiIntegration::new();
    // Create renderer (Gui integration will be initialized there)
    let mut renderer = VulkanoWinitRenderer::new(
        &event_loop,
        1280,
        720,
        PresentMode::Immediate,
        "Basic Example",
        &mut gui,
    );
    // Create gui state (this should occur after renderer so we have access to gfx queue etc.)
    let mut gui_state = AppGuiState::new(&mut gui);
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
                // It's a closure inside which you can call anything. Here we're calling the layout
                // of our `gui_state` which will have mutable access to our gui integration
                gui.immediate_ui(|ctx| gui_state.layout(ctx));
                // Lastly we'll render our ui
                renderer.render(&mut gui);
            }
            Event::MainEventsCleared => {
                renderer.window().request_redraw();
            }
            _ => (),
        }
    });
}
