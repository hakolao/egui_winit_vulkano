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
let mut gui = Gui::new(renderer.surface(), renderer.queue(), false);
// Or with subpass
let mut gui = Gui::new_with_subpass(renderer.surface(), renderer.queue(), subpass);
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
renderer.render(&mut gui); //... and inside render function:
// Draw, where
// future = acquired future from previous_frame_end.join(swapchain_acquire_future) and
// image_view_to_draw_on = the final image onto which you wish to render UI, usually e.g.
// self.final_images[image_num].clone() = one of your swap chain images.
let after_future = gui.draw_on_image(future, image_view_to_draw_on);

// Or if you created the integration with subpass
let cb = gui.draw_on_subpass_image(framebuffer_dimensions);
draw_pass.execute(cb);
```
6. Finish your render by waiting on the future `gui.draw` returns. See `finish` function in example renderers.
Or in the case of subpass, execute your commands.

See the examples directory for a more wholesome example which uses Vulkano developers' frame system to organize rendering.

Remember, on Linux, you need to install following to run Egui
```bash
sudo apt-get install libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev
```

# Examples

```sh
cargo run --example wholesome
cargo run --example minimal
cargo run --example subpass
```

# Notes
This integration would not have been possible without the examples from [vulkano-examples](https://github.com/vulkano-rs/vulkano/tree/master/examples/src/bin)
or [egui_winit_ash_vk_mem](https://github.com/MatchaChoco010/egui_winit_ash_vk_mem).
