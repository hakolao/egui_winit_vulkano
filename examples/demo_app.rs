// Copyright (c) 2021 Okko Hakola
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use egui_winit_vulkano::Gui;
use vulkano_util::{
    context::{VulkanoConfig, VulkanoContext},
    window::{VulkanoWindows, WindowDescriptor},
};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

// Simply create egui demo apps to test everything works correctly.
// Creates two windows with different color formats for their swapchain.

pub fn main() {
    // Winit event loop
    let event_loop = EventLoop::new();
    // Vulkano context
    let context = VulkanoContext::new(VulkanoConfig::default());
    // Vulkano windows (create one)
    let mut windows = VulkanoWindows::default();
    let window1 =
        windows.create_window(&event_loop, &context, &WindowDescriptor::default(), |ci| {
            ci.image_format = Some(vulkano::format::Format::B8G8R8A8_SRGB)
        });
    let window2 =
        windows.create_window(&event_loop, &context, &WindowDescriptor::default(), |ci| {
            ci.image_format = Some(vulkano::format::Format::B8G8R8A8_UNORM)
        });
    // Create gui as main render pass (no overlay means it clears the image each frame)
    let mut gui1 = {
        let renderer = windows.get_renderer_mut(window1).unwrap();
        Gui::new(
            renderer.surface(),
            Some(renderer.swapchain_format()),
            renderer.graphics_queue(),
            false,
        )
    };
    let mut gui2 = {
        let renderer = windows.get_renderer_mut(window2).unwrap();
        Gui::new(
            renderer.surface(),
            Some(renderer.swapchain_format()),
            renderer.graphics_queue(),
            false,
        )
    };
    // Display the demo application that ships with egui.
    let mut demo_app1 = egui_demo_lib::DemoWindows::default();
    let mut demo_app2 = egui_demo_lib::DemoWindows::default();
    let mut egui_test1 = egui_demo_lib::ColorTest::default();
    let mut egui_test2 = egui_demo_lib::ColorTest::default();

    event_loop.run(move |event, _, control_flow| {
        for (wi, renderer) in windows.iter_mut() {
            // Quick and ugly...
            let gui = if *wi == window1 { &mut gui1 } else { &mut gui2 };
            let demo_app = if *wi == window1 { &mut demo_app1 } else { &mut demo_app2 };
            let egui_test = if *wi == window1 { &mut egui_test1 } else { &mut egui_test2 };
            match &event {
                Event::WindowEvent { event, window_id } if window_id == wi => {
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
                            *control_flow = ControlFlow::Exit;
                        }
                        _ => (),
                    }
                }
                Event::RedrawRequested(window_id) if window_id == wi => {
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
                    let before_future = renderer.acquire().unwrap();
                    // Render gui
                    let after_future =
                        gui.draw_on_image(before_future, renderer.swapchain_image_view());
                    // Present swapchain
                    renderer.present(after_future, true);
                }
                Event::MainEventsCleared => {
                    renderer.surface().window().request_redraw();
                }
                _ => (),
            }
        }
    });
}
