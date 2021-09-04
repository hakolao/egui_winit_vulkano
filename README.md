# egui_winit_vulkano

![MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Apache](https://img.shields.io/badge/license-Apache-blue.svg)

This is an [egui](https://github.com/emilk/egui) integration for
[winit](https://github.com/rust-windowing/winit) and [vulkano](https://github.com/vulkano-rs/vulkano).

You'll need a Vulkano target image as an input to which the UI will be painted.
The aim of this is to allow a simple enough API to separate UI nicely out of your renderer and make it easy to build your immediate mode UI with Egui.

# Usage
1. Create your own renderer with Vulkano, and allow access to Vulkano's gfx queue `Arc<Queue>` and Vulkano's winit surface `Arc<Surface<Window>>`
2. Create Gui integration with the surface & gfx queue
```rust
// Has its own renderpass
let mut gui = Gui::new(renderer.surface(), renderer.queue());
// Or with subpass
let mut gui = Gui::new_with_subpass(renderer.surface(), renderer.queue(), subpass);
```
3. Inside your event loop, update `gui` integration
```rust
event_loop.run(move |event, _, control_flow| {
    // Update Egui integration so the UI works!
    gui.update(&event);
    // ...match event {..}
});
```
4. Fill immediate mode UI through the integration in `Event::RedrawRequested` before you render
```rust
gui.immediate_ui(|gui| {
    let ctx = gui.context();
    // Fill egui UI layout here
    // It may be convenient to organize the layout under a stateful GuiState struct (See `wholesome` example)
});
```
5. Render gui via your renderer on any image or most likely on your swapchain images:
```rust
renderer.render(&mut gui); //... and inside render function:
// Draw, where
// future = acquired future from previous_frame_end.join(swapchain_acquire_future) and
// image_view_to_draw_on = the final image onto which you wish to render UI, usually e.g.
// self.final_images[image_num].clone() = one of your swap chain images.
// [0.0, 0.0, 0.0, 0.0] = clear color
let after_future = gui.draw_on_image(future, image_view_to_draw_on, [0.0, 0.0, 0.0, 0.0]);

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
