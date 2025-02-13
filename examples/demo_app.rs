// Copyright (c) 2021 Okko Hakola
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use egui_demo_lib::{ColorTest, DemoWindows};
use egui_winit_vulkano::{Gui, GuiConfig};
use vulkano_util::{
    context::{VulkanoConfig, VulkanoContext},
    window::{VulkanoWindows, WindowDescriptor},
};
use winit::{
    application::ApplicationHandler, error::EventLoopError, event::WindowEvent,
    event_loop::EventLoop, window::WindowId,
};

// Simply create egui demo apps to test everything works correctly.
// Creates two windows with different color formats for their swapchain.

pub struct App {
    context: VulkanoContext,
    windows: VulkanoWindows,
    window1: Option<Window>,
    window2: Option<Window>,
}

pub struct Window {
    id: WindowId,
    gui: Gui,
    demo_app: DemoWindows,
    egui_test: ColorTest,
}

impl Default for App {
    fn default() -> Self {
        // Vulkano context
        let context = VulkanoContext::new(VulkanoConfig::default());
        // Vulkano windows (create one)
        let windows = VulkanoWindows::default();

        Self { context, windows, window1: None, window2: None }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        // Display the demo application that ships with egui.
        let demo_app1 = DemoWindows::default();
        let demo_app2 = DemoWindows::default();
        let egui_test1 = ColorTest::default();
        let egui_test2 = ColorTest::default();

        let window1 = self.windows.create_window(
            event_loop,
            &self.context,
            &WindowDescriptor {
                title: String::from("egui_winit_vulkano SRGB"),
                ..WindowDescriptor::default()
            },
            |ci| {
                ci.image_format = vulkano::format::Format::B8G8R8A8_SRGB;
                ci.min_image_count = ci.min_image_count.max(2);
            },
        );

        let window2 = self.windows.create_window(
            event_loop,
            &self.context,
            &WindowDescriptor {
                title: String::from("egui_winit_vulkano UNORM"),
                ..WindowDescriptor::default()
            },
            |ci| {
                ci.image_format = vulkano::format::Format::B8G8R8A8_UNORM;
                ci.min_image_count = ci.min_image_count.max(2);
            },
        );

        // Create gui as main render pass (no overlay means it clears the image each frame)
        let gui1 = {
            let renderer = self.windows.get_renderer_mut(window1).unwrap();
            Gui::new(
                event_loop,
                renderer.surface(),
                renderer.graphics_queue(),
                renderer.swapchain_format(),
                GuiConfig { allow_srgb_render_target: true, ..GuiConfig::default() },
            )
        };

        let gui2 = {
            let renderer = self.windows.get_renderer_mut(window2).unwrap();
            Gui::new(
                event_loop,
                renderer.surface(),
                renderer.graphics_queue(),
                renderer.swapchain_format(),
                GuiConfig::default(),
            )
        };

        self.window1 =
            Some(Window { id: window1, gui: gui1, demo_app: demo_app1, egui_test: egui_test1 });

        self.window2 =
            Some(Window { id: window2, gui: gui2, demo_app: demo_app2, egui_test: egui_test2 });
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let renderer = self.windows.get_renderer_mut(window_id).unwrap();

        let w1 = self.window1.as_mut().unwrap();
        let w2 = self.window2.as_mut().unwrap();

        // Quick and ugly...
        let gui = if window_id == w1.id { &mut w1.gui } else { &mut w2.gui };
        let demo_app = if window_id == w1.id { &mut w1.demo_app } else { &mut w2.demo_app };
        let egui_test = if window_id == w1.id { &mut w1.egui_test } else { &mut w2.egui_test };

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
                // Set immediate UI in redraw here
                gui.immediate_ui(|gui| {
                    let ctx = gui.context();
                    demo_app.ui(&ctx);

                    egui::Window::new("Colors").vscroll(true).show(&ctx, |ui| {
                        egui_test.ui(ui);
                    });
                });
                // Alternatively you could
                // gui.begin_frame();
                // let ctx = gui.context();
                // demo_app.ui(&ctx);

                // Render UI
                // Acquire swapchain future
                match renderer.acquire(Some(std::time::Duration::from_millis(10)), |_| {}) {
                    Ok(future) => {
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
        for (_, renderer) in self.windows.iter_mut() {
            renderer.window().request_redraw();
        }
    }
}

pub fn main() -> Result<(), EventLoopError> {
    // Winit event loop
    let event_loop = EventLoop::new().unwrap();

    let mut app = App::default();

    event_loop.run_app(&mut app)
}
