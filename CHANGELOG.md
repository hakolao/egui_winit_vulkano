# Changelog

## v0.18.0
- Use `vulkano::format::Format::B8G8R8A8_SRGB` and fix shaders to get the right color output. This means that
the swapchain images should be created with the same `format`.
- Add `demo_app` example with `color_test`
- Use `ImmutableImage` for font images rather than Vulkano's `StorageImage`.