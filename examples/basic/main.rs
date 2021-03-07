use egui::{CtxRef, Visuals};
use egui_winit_vulkano::Gui;
use vulkano::swapchain::PresentMode;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

use crate::{renderer::VulkanoWinitRenderer, time_info::TimeInfo};

mod frame_system;
mod renderer;
mod time_info;

/// Example struct to contain the state of the UI
pub struct AppGuiState {
    show_texture_window: bool,
    image_texture_id: egui::TextureId,
}

impl AppGuiState {
    pub fn new(gui: &mut Gui) -> AppGuiState {
        let image_texture_id = gui.register_user_image(include_bytes!("./assets/tree.png"));
        AppGuiState { show_texture_window: false, image_texture_id }
    }

    /// Defines the layout of our UI
    pub fn layout(&mut self, egui_context: CtxRef, window: &Window, fps: f32) {
        egui_context.set_visuals(Visuals::dark());
        egui::SidePanel::left("Side Panel", 150.0).show(&egui_context, |ui| {
            ui.heading("Hello Tree");
            ui.separator();
            ui.checkbox(&mut self.show_texture_window, "Show Tree");
        });
        let show_texture_window = &mut self.show_texture_window;
        let image_texture_id = self.image_texture_id;
        egui::Window::new("Mah Tree").resizable(true).scroll(true).open(show_texture_window).show(
            &egui_context,
            |ui| {
                ui.image(image_texture_id, [256.0, 256.0]);
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
    // Create gui integration shell
    let mut gui = Gui::new();
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
                // It's a closure giving access to egui contexe inside which you can call anything.
                // Here we're calling the layout of our `gui_state`.
                gui.immediate_ui(|ctx| gui_state.layout(ctx, renderer.window(), time.fps()));
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
