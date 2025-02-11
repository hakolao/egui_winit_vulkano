// Copyright (c) 2021 Okko Hakola
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::collections::HashMap;

use egui_winit_vulkano::{Gui, GuiConfig};
use vulkano_util::{
    context::{VulkanoConfig, VulkanoContext},
    window::{VulkanoWindows, WindowDescriptor},
};
use winit::{
    application::ApplicationHandler, event::WindowEvent, event_loop::EventLoop, window::WindowId,
};

// Simply create egui demo apps to test everything works correctly.
// Creates two windows with different color formats for their swapchain.

fn main() {
    // Winit event loop
    let event_loop = EventLoop::new().unwrap();
    event_loop
        .run_app(&mut App {
            // Vulkano context
            context: VulkanoContext::new(VulkanoConfig::default()),
            // Vulkano windows (create one)
            windows: VulkanoWindows::default(),
            window_context: HashMap::new(),
        })
        .unwrap();
}

struct App {
    context: VulkanoContext,
    windows: VulkanoWindows,
    window_context: HashMap<WindowId, WindowContext>,
}

struct WindowContext {
    gui: Gui,
    demo_app: egui_demo_lib::DemoWindows,
    egui_test: egui_demo_lib::ColorTest,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        // Vulkano windows (create one)
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

        self.window_context.insert(
            window1,
            WindowContext {
                gui: gui1,
                demo_app: egui_demo_lib::DemoWindows::default(),
                egui_test: egui_demo_lib::ColorTest::default(),
            },
        );
        self.window_context.insert(
            window2,
            WindowContext {
                gui: gui2,
                demo_app: egui_demo_lib::DemoWindows::default(),
                egui_test: egui_demo_lib::ColorTest::default(),
            },
        );
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let wc = self.window_context.get_mut(&window_id).unwrap();
        let renderer = self.windows.get_renderer_mut(window_id).unwrap();

        // Update Egui integration so the UI works!
        let _pass_events_to_game = !wc.gui.update(renderer.window(), &event);
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
                wc.gui.immediate_ui(|gui| {
                    let ctx = gui.context();
                    wc.demo_app.ui(ctx);

                    egui::Window::new("Colors").vscroll(true).show(ctx, |ui| {
                        wc.egui_test.ui(ui);
                    });
                });
                // Alternatively you could
                // gui.begin_frame();
                // let ctx = gui.context();
                // demo_app.ui(&ctx);

                // Render UI
                // Acquire swapchain future
                match renderer.acquire(None, |_| {}) {
                    Ok(future) => {
                        let after_future =
                            wc.gui.draw_on_image(future, renderer.swapchain_image_view());
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
        for (_, renderer) in self.windows.iter() {
            renderer.window().request_redraw();
        }
    }
}
