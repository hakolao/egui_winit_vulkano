// Copyright (c) 2021 Okko Hakola
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

#![allow(clippy::eq_op)]

use egui::{ScrollArea, TextEdit, TextStyle};
use egui_winit_vulkano::{Gui, GuiConfig};
use vulkano_util::{
    context::{VulkanoConfig, VulkanoContext},
    window::{VulkanoWindows, WindowDescriptor},
};
use winit::{
    application::ApplicationHandler, error::EventLoopError, event::WindowEvent,
    event_loop::EventLoop,
};

fn sized_text(ui: &mut egui::Ui, text: impl Into<String>, size: f32) {
    ui.label(egui::RichText::new(text).size(size));
}

pub struct App {
    context: VulkanoContext,
    windows: VulkanoWindows,
    code: String,
    gui: Option<Gui>,
}

impl Default for App {
    fn default() -> Self {
        // Vulkano context
        let context = VulkanoContext::new(VulkanoConfig::default());

        // Vulkano windows
        let windows = VulkanoWindows::default();

        let code = CODE.to_owned();

        Self { context, windows, code, gui: None }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        self.windows.create_window(event_loop, &self.context, &WindowDescriptor::default(), |ci| {
            ci.image_format = vulkano::format::Format::B8G8R8A8_UNORM;
            ci.min_image_count = ci.min_image_count.max(2);
        });

        // Create gui as main render pass (no overlay means it clears the image each frame)
        self.gui = Some({
            let renderer = self.windows.get_primary_renderer_mut().unwrap();
            Gui::new(
                event_loop,
                renderer.surface(),
                renderer.graphics_queue(),
                renderer.swapchain_format(),
                GuiConfig::default(),
            )
        });
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let renderer = self.windows.get_renderer_mut(window_id).unwrap();

        let gui = self.gui.as_mut().unwrap();

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
                // Set immediate UI in redraw here
                gui.immediate_ui(|gui| {
                    let ctx = gui.context();
                    egui::CentralPanel::default().show(&ctx, |ui| {
                        ui.vertical_centered(|ui| {
                            ui.add(egui::widgets::Label::new("Hi there!"));
                            sized_text(ui, "Rich Text", 32.0);
                        });
                        ui.separator();
                        ui.columns(2, |columns| {
                            ScrollArea::vertical().id_salt("source").show(&mut columns[0], |ui| {
                                ui.add(
                                    TextEdit::multiline(&mut self.code).font(TextStyle::Monospace),
                                );
                            });
                            ScrollArea::vertical().id_salt("rendered").show(
                                &mut columns[1],
                                |ui| {
                                    egui_demo_lib::easy_mark::easy_mark(ui, &self.code);
                                },
                            );
                        });
                    });
                });
                // Render UI
                // Acquire swapchain future
                match renderer.acquire(Some(std::time::Duration::from_millis(10)), |_| {}) {
                    Ok(future) => {
                        // Render gui
                        let after_future = self
                            .gui
                            .as_mut()
                            .unwrap()
                            .draw_on_image(future, renderer.swapchain_image_view());
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
        let renderer = self.windows.get_primary_renderer_mut().unwrap();
        renderer.window().request_redraw();
    }
}

pub fn main() -> Result<(), EventLoopError> {
    // Winit event loop
    let event_loop = EventLoop::new().unwrap();

    let mut app = App::default();

    event_loop.run_app(&mut app)
}

const CODE: &str = r"
# Some markup
```
let mut gui = Gui::new(&event_loop, renderer.surface(), None, renderer.queue(), SampleCount::Sample1);
```
";
