# egui_winit_vulkano

[![Crates.io](https://img.shields.io/crates/v/egui_winit_vulkano.svg)](https://crates.io/crates/egui_winit_vulkano)
![Apache](https://img.shields.io/badge/license-Apache-blue.svg)
![CI](https://github.com/hakolao/egui_winit_vulkano/workflows/CI/badge.svg)

This is an [egui](https://github.com/emilk/egui) integration for
[winit](https://github.com/rust-windowing/winit) and [vulkano](https://github.com/vulkano-rs/vulkano).

You'll need a Vulkano target image as an input to which the UI will be painted.
The aim of this is to allow a simple enough API to separate UI nicely out of your renderer and make it easy to build your immediate mode UI with Egui.

# Usage
1. Create your own renderer with Vulkano, and allow access to Vulkano's gfx queue `Arc<Queue>` and Vulkano's winit surface `Arc<Surface<Window>>`
2. Create Gui integration with the surface & gfx queue

```rust
// Has its own renderpass (is_overlay = false means that the renderpass will clear the image, true means
// that the caller is responsible for clearing the image
let mut gui = Gui::new(renderer.surface(), None, renderer.queue(), false);
// Or with subpass. This means that you must create the renderpass yourself. Egui subpass will then draw on your
// image.
let mut gui = Gui::new_with_subpass(renderer.surface(), None, renderer.queue(), subpass);
```

3. Inside your event loop, update `gui` integration with `WindowEvent`

```rust
gui.update(&event);
```

4. Fill immediate mode UI through the integration in `Event::RedrawRequested` before you render
```rust
gui.immediate_ui(|gui| {
    let ctx = gui.context();
    // Fill egui UI layout here
});

// Or

gui.begin_frame();
// fill egui layout...


// And when you render with `gui.draw_on_image(..)`, this will finish the egui frame
```
5. Render gui via your renderer on any image or most likely on your swapchain images:
```rust
// Acquire swapchain future
let before_future = renderer.acquire().unwrap();
// Render gui by passing the acquire future (or any) and render target image (swapchain image view)
let after_future = gui.draw_on_image(before_future, renderer.swapchain_image_view());
// Present swapchain
renderer.present(after_future, true);
// ----------------------------------
// Or if you created the integration with subpass
let cb = gui.draw_on_subpass_image(framebuffer_dimensions);
draw_pass.execute(cb);
```
See the examples directory for better usage guidance.

Remember, on Linux, you need to install following to run Egui
```bash
sudo apt-get install libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev
```

# Examples

```sh
cargo run --example wholesome
cargo run --example minimal
cargo run --example subpass
cargo run --example demo_app
```

# Notes
This integration would not have been possible without the examples from [vulkano-examples](https://github.com/vulkano-rs/vulkano/tree/master/examples/src/bin)
or [egui_winit_ash_vk_mem](https://github.com/MatchaChoco010/egui_winit_ash_vk_mem).
