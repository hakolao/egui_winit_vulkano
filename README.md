# egui_winit_vulkano

![MIT](https://img.shields.io/badge/license-MIT-blue.svg)

This is the [egui](https://github.com/emilk/egui) integration for
[winit](https://github.com/rust-windowing/winit) and [vulkano](https://github.com/vulkano-rs/vulkano).

# Usage

1. Create a renderer with Vulkano, and allow access to the inputs needed for `egui_winit_vulkano`.
   The contents of the renderer are for you to decide, but allow access to at least parameters listed below.
2. Create integration:

```rs
let mut gui = egui_winit_vulkano::Gui::new(
    renderer.window().inner_size(), // Winit window's size
    renderer.window().scale_factor(), // Winit's scale factor
    renderer.queue(), // Vulkano's Queue (Arc<Queue>)
    renderer.deferred_subpass(), // Vulkano's render pass' subpass
);
```
3. Inside your renderer, call `gui.draw` before presenting the swapchain.
   Draw will render egui meshes each frame using your renderer's subpass
```rs
gui.draw(
    self.surface.window(), // Pass winit window to gui
    draw_pass.viewport_dimensions() // Pass view port dimensions [u32; 2]
);
```
4. Fill the immediate mode UI in your event loop in `main.rs` or wherever your loop resides:
```rs
Event::RedrawRequested(window_id) if window_id == window_id => {
   // In this closure, organize your UI layout
    gui.immediate_ui(|ctx| {
        // Add UI here. See egui's docs & examples folder
    });
    // Lastly we'll render our ui
    renderer.render(&mut gui);
}
```

See the examples directory for a more wholesome example which uses Vulkano developers' frame system to organize rendering.

Remember, on Linux, you need to install following to run egui
```bash
sudo apt-get install libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev
```

# Example

```sh
cargo run --example wholesome
```