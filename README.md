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
// Has its own renderpass. Modify GuiConfig to determine image clear behavior etc.
let mut gui = Gui::new(&event_loop, renderer.surface(), renderer.queue(), renderer.swapchain_format(), GuiConfig::default());
// Or with subpass. This means that you must create the renderpass yourself. Egui subpass will then draw on your
// image.
let mut gui = Gui::new_with_subpass(&event_loop, renderer.surface(), renderer.queue(), renderer.swapchain_format(), subpass, GuiConfig::default());
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
Note that Egui strongly prefers UNORM render targets, and passing an sRGB color space image is considered an error. See [The correct color space](#the-correct-color-space) for more details on how to deal with it.

See the examples directory for better usage guidance.

Remember, on Linux, you need to install following to run Egui
```bash
sudo apt-get install libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev
```

# Examples

```sh
./run_all_examples.sh
```

# The correct color space
The key takeaway is that egui assumes any math in the shaders as well as alpha blending is done linearly, but using colors in sRGB color space. This is great for platforms which only have linear textures, like webgl, webgpu etc. This means for vulkan you must also use UNORM rendertargets, into which the shader will write color values in sRGB color space. This is highly inconvenient to work with, and there are several approaches around this issue:

1. The simplest approach would be to create the swapchain using a UNORM image format, as is done in most examples. However, the swapchain image will be displayed in sRGB color space, so any normal rendering operations should write to an sRGB image, otherwise the color you emit in your fragment shader will not match the color displayed. This makes this approach only viable if you only want to draw things using Egui.
2. You render into a UNORM image you allocated, and then copy that image onto your swapchain image using compute shaders or a fullscreen triangle. Your shader needs to read the UNORM image, which if you remember contains values in sRGB color space, convert it from sRGB to linear color space, and then write it to your sRGB swapchain image. This conversion is also why you cannot just blit the image, but need to use a shader in between.
3. You again render into an image you allocated, but this time with `VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT` flag (the format may be UNORM or SRGB). This flag allows you to create image views with [compatible but differing formats](https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#formats-compatibility-classes). Create two image views, one viewing your data as UNORM and one as sRGB (e.g. `VK_FORMAT_R8G8B8A8_UNORM` and `VK_FORMAT_R8G8B8A8_SRGB`) by modifying `ImageViewCreateInfo::format` parameter from the default value. Render your Egui into the UNORM image view, then you may blit it from the sRGB image view onto your framebuffer.
4. If your device supports [`VK_KHR_swapchain_mutable_format`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_swapchain_mutable_format.html), which AMD, Nvidia and Intel do on both Windows and Linux, you may get around having to allocate an extra image and blit it completely. Enabling this extension and passing the `VK_SWAPCHAIN_CREATE_MUTABLE_FORMAT_BIT_KHR` flag when creating your swapchain, which should be in sRGB format, allows you to create image views of your swapchain with differing formats. So just like in 2, you may now create two image views, one as sRGB to use in your 3D rendering, and one as UNORM to use in a following Egui pass. But as you are rendering directly into your swapchain, there is no need for copying or blitting.

For desktop platforms, we would recommend approach 4 as it's supported by all major platforms and vendors, the most convenient to use, and saves both memory and memory bandwidth. If you intend to go for maximum compatibility, we recommend implementing approach 3 as it does not require any extra features or extensions, while still supporting approach 4 as it should be trivial to implement. Approach 2 is only interesting if you can combine the in-shader copy with some other post-processing shaders you would have to run anyway.

If you so wish, you may still draw to an image view in sRGB format, if you can accept discolorations when elements are alpha blended together. Doing so requires enabling `GuiConfig::allow_srgb_render_target`, as otherwise it is an error to draw to an sRGB image view. Normally, you would expect that drawing to the wrong color space will cause the entire UI to be discolored. But enabling that option also slightly changes the shader we are using for drawing UI elements, to only have discolorations in alpha blended areas instead of the entire image. 

# Notes
This integration would not have been possible without the examples from [vulkano-examples](https://github.com/vulkano-rs/vulkano/tree/master/examples/src/bin)
or [egui_winit_ash_vk_mem](https://github.com/MatchaChoco010/egui_winit_ash_vk_mem).
