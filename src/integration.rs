// Copyright (c) 2021 Okko Hakola
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.
use std::sync::Arc;

use vulkano::{
    command_buffer::SecondaryAutoCommandBuffer, device::Queue, image::ImageViewAbstract,
    render_pass::Subpass, swapchain::Surface, sync::GpuFuture,
};
use winit::{event::Event, window::Window};

use crate::{
    context::Context,
    renderer::Renderer,
    utils::{texture_from_bytes, texture_from_file},
};

pub struct Gui {
    context: Context,
    renderer: Renderer,
    surface: Arc<Surface<Window>>,
}

impl Gui {
    /// Creates new Egui to Vulkano integration by setting the necessary parameters
    /// This is to be called once we have access to vulkano_win's winit window surface
    /// and gfx queue. Created with this, the renderer will own a render pass which is useful to e.g. place your render pass' images
    /// onto egui windows
    /// - `surface`: Vulkano's Winit Surface [`Arc<Surface<Window>>`]
    /// - `gfx_queue`: Vulkano's [`Queue`]
    /// - `is_overlay`: If true, you should be responsible for clearing the image before `draw_on_image`, else it gets cleared
    pub fn new(surface: Arc<Surface<Window>>, gfx_queue: Arc<Queue>, is_overlay: bool) -> Gui {
        let format = gfx_queue
            .device()
            .physical_device()
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0;
        let context = Context::new(surface.window().inner_size(), surface.window().scale_factor());
        let renderer = Renderer::new_with_render_pass(gfx_queue, format, is_overlay);
        Gui { context, renderer, surface }
    }

    /// Same as `new` but instead of integration owning a render pass, egui renders on your subpass
    pub fn new_with_subpass(
        surface: Arc<Surface<Window>>,
        gfx_queue: Arc<Queue>,
        subpass: Subpass,
    ) -> Gui {
        let format = gfx_queue
            .device()
            .physical_device()
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0;
        let context = Context::new(surface.window().inner_size(), surface.window().scale_factor());
        let renderer = Renderer::new_with_subpass(gfx_queue, format, subpass);
        Gui { context, renderer, surface }
    }

    /// Updates context state by winit event.
    pub fn update<T>(&mut self, winit_event: &Event<T>) {
        self.context.handle_event(winit_event)
    }

    /// Sets Egui integration's UI layout. This must be called before draw
    /// Begins Egui frame
    pub fn immediate_ui(&mut self, layout_function: impl FnOnce(&mut Self)) {
        self.context.begin_frame();
        // Render Egui
        layout_function(self);
    }

    /// If you wish to better control when to begin frame, do so by calling this function
    /// (Finish by drawing)
    pub fn begin_frame(&mut self) {
        self.context.begin_frame();
    }

    /// Renders ui on `final_image` & Updates cursor icon
    /// Finishes Egui frame
    /// - `before_future` = Vulkano's GpuFuture
    /// - `final_image` = Vulkano's image (render target)
    pub fn draw_on_image<F>(
        &mut self,
        before_future: F,
        final_image: Arc<dyn ImageViewAbstract + 'static>,
    ) -> Box<dyn GpuFuture>
    where
        F: GpuFuture + 'static,
    {
        if !self.renderer.has_renderpass() {
            panic!(
                "Gui integration has been created with subpass, use `draw_on_subpass_image` \
                 instead"
            )
        }
        let (output, clipped_meshes) = self.context.end_frame();
        self.context.update_cursor_icon(self.surface.window(), output.cursor_icon);
        // Draw egui meshes
        self.renderer.draw_on_image(&mut self.context, clipped_meshes, before_future, final_image)
    }

    /// Creates commands for rendering ui on subpass' image and returns the command buffer for execution on your side
    /// Finishes Egui frame
    /// - `final_image` = Vulkano's image (render target)
    pub fn draw_on_subpass_image(
        &mut self,
        image_dimensions: [u32; 2],
    ) -> SecondaryAutoCommandBuffer {
        if self.renderer.has_renderpass() {
            panic!(
                "Gui integration has been created with its own render pass, use `draw_on_image` \
                 instead"
            )
        }
        let (output, clipped_meshes) = self.context.end_frame();
        self.context.update_cursor_icon(self.surface.window(), output.cursor_icon);
        // You still need to execute that command buffer yourself
        self.renderer.draw_on_subpass_image(&mut self.context, clipped_meshes, image_dimensions)
    }

    /// Registers a user image from Vulkano image view to be used by egui
    pub fn register_user_image_view(
        &mut self,
        image: Arc<dyn ImageViewAbstract + Send + Sync>,
    ) -> egui::TextureId {
        self.renderer.register_user_image(image)
    }

    /// Registers a user image to be used by egui
    /// - `image_file_bytes`: e.g. include_bytes!("./assets/tree.png")
    /// - `format`: e.g. vulkano::format::Format::R8G8B8A8Unorm
    pub fn register_user_image(
        &mut self,
        image_file_bytes: &[u8],
        format: vulkano::format::Format,
    ) -> egui::TextureId {
        let image = texture_from_file(self.renderer.queue(), image_file_bytes, format)
            .expect("Failed to create image");
        self.renderer.register_user_image(image)
    }

    pub fn register_user_image_from_bytes(
        &mut self,
        image_byte_data: &[u8],
        dimensions: (u64, u64),
        format: vulkano::format::Format,
    ) -> egui::TextureId {
        let image = texture_from_bytes(self.renderer.queue(), image_byte_data, dimensions, format)
            .expect("Failed to create image");
        self.renderer.register_user_image(image)
    }

    /// Unregisters a user image
    pub fn unregister_user_image(&mut self, texture_id: egui::TextureId) {
        self.renderer.unregister_user_image(texture_id);
    }

    /// Access egui's context (which can be used to e.g. set fonts, visuals etc)
    pub fn context(&self) -> egui::CtxRef {
        self.context.context()
    }
}
