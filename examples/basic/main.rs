use vulkano::swapchain::PresentMode;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

use crate::renderer::VulkanoWinitRenderer;

mod frame_system;
mod renderer;

pub fn main() {
    let event_loop = EventLoop::new();
    let mut renderer =
        VulkanoWinitRenderer::new(&event_loop, 1280, 720, PresentMode::Immediate, "Basic Example");
    event_loop.run(move |event, _, control_flow| {
        // Update Egui Context state
        renderer.egui_update(&event);
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
                renderer.render();
            }
            Event::MainEventsCleared => {
                renderer.window().request_redraw();
            }
            _ => (),
        }
    });
}
