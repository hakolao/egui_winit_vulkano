use std::sync::Arc;

use egui::CtxRef;
use vulkano::{
    command_buffer::AutoCommandBuffer,
    device::Queue,
    framebuffer::{RenderPassAbstract, Subpass},
    image::ImageViewAccess,
};
use winit::{dpi::PhysicalSize, event::Event, window::Window};

use crate::{context::EguiContext, renderer::EguiVulkanoRenderer, utils::texture_from_file_bytes};

pub struct Gui {
    context: Option<EguiContext>,
    renderer: Option<EguiVulkanoRenderer>,
}

impl Gui {
    /// Instantiates a new integration struct as an empty shell
    pub fn new() -> Gui {
        Gui { context: None, renderer: None }
    }

    /// Initializes Egui to Vulkano integration by setting the necessary parameters
    /// This is to be called once we have access to vulkano_win's winit window surface
    /// and after render pass has been created
    /// - `size`: Size of the window as [PhysicalSize<u32>]
    /// - `scale_factor`: pointes per pixel, = `window.scale_factor()`
    /// - `gfx_queue`: Vulkano's [`Queue`]
    /// - `subpass`: Vulkano's subpass created from render pass, see examples
    /// - Render pass must have depth attachment and at least one color attachment
    pub fn init<R>(
        &mut self,
        size: PhysicalSize<u32>,
        scale_factor: f64,
        gfx_queue: Arc<Queue>,
        subpass: Subpass<R>,
    ) where
        R: RenderPassAbstract + Send + Sync + 'static,
    {
        assert!(subpass.has_depth());
        assert!(subpass.num_color_attachments() >= 1);
        // ToDo: Validate what ever is useful
        self.context = Some(EguiContext::new(size, scale_factor));
        self.renderer = Some(EguiVulkanoRenderer::new(gfx_queue.clone(), subpass));
    }

    /// Updates context state by winit event. Integration must have been initialized
    pub fn update<T>(&mut self, winit_event: &Event<T>) {
        assert!(self.context.is_some() && self.renderer.is_some());
        self.context.as_mut().unwrap().handle_event(winit_event)
    }

    /// Sets Egui integration's UI layout. This must be called before draw
    /// Begins Egui frame
    pub fn immediate_ui(&mut self, layout_function: impl FnOnce(CtxRef)) {
        assert!(self.context.is_some() && self.renderer.is_some());
        self.context.as_mut().unwrap().begin_frame();
        // Render Egui
        layout_function(self.context());
    }

    /// Renders ui & Updates cursor icon
    /// Finishes Egui frame
    pub fn draw(&mut self, window: &Window, framebuffer_dimensions: [u32; 2]) -> AutoCommandBuffer {
        assert!(self.context.is_some() && self.renderer.is_some());
        // Get outputs of `immediate_ui`
        let (output, clipped_meshes) = self.context.as_mut().unwrap().end_frame();
        // Update cursor icon
        self.context.as_mut().unwrap().update_cursor_icon(window, output.cursor_icon);
        // Draw egui meshes
        let cb = self.renderer.as_mut().unwrap().draw(
            self.context.as_mut().unwrap(),
            clipped_meshes,
            framebuffer_dimensions,
        );
        cb
    }

    /// Registers a user image from Vulkano image view to be used by egui
    pub fn register_user_image_view(
        &mut self,
        image: Arc<dyn ImageViewAccess + Send + Sync>,
    ) -> egui::TextureId {
        assert!(self.context.is_some() && self.renderer.is_some());
        self.renderer.as_mut().unwrap().register_user_image(image)
    }

    /// Registers a user image to be used by egui
    /// - `image_file_bytes`: e.g. include_bytes!("./assets/tree.png")
    pub fn register_user_image(&mut self, image_file_bytes: &[u8]) -> egui::TextureId {
        assert!(self.context.is_some() && self.renderer.is_some());
        let image =
            texture_from_file_bytes(self.renderer.as_ref().unwrap().queue(), image_file_bytes)
                .expect("Failed to create image");
        self.renderer.as_mut().unwrap().register_user_image(image)
    }

    /// Unregisters a user image
    pub fn unregister_user_image(&mut self, texture_id: egui::TextureId) {
        assert!(self.context.is_some() && self.renderer.is_some());
        self.renderer.as_mut().unwrap().unregister_user_image(texture_id);
    }

    /// Access egui's context (which can be used to e.g. set fonts, visuals etc)
    pub fn context(&self) -> egui::CtxRef {
        self.context.as_ref().unwrap().context()
    }
}
