use std::sync::Arc;

use vulkano::{
    command_buffer::AutoCommandBuffer,
    device::Queue,
    framebuffer::{RenderPassAbstract, Subpass},
    image::ImageViewAccess,
};
use winit::{dpi::PhysicalSize, event::Event, window::Window};

use crate::{EguiContext, EguiVulkanoRenderer};

pub struct EguiIntegration {
    context: Option<EguiContext>,
    renderer: Option<EguiVulkanoRenderer>,
    layout: Option<fn()>,
}

impl EguiIntegration {
    /// Instantiates a new integration struct as an empty shell
    pub fn new() -> EguiIntegration {
        EguiIntegration { context: None, renderer: None, layout: None }
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

    /// Sets Egui integration's UI layout. This should be called after
    pub fn set_layout(&mut self, layout_function: fn()) {
        assert!(self.context.is_some() && self.renderer.is_some());
        self.layout = Some(layout_function);
    }

    /// Renders ui & Updates cursor icon
    pub fn draw(&mut self, window: &Window, framebuffer_dimensions: [u32; 2]) -> AutoCommandBuffer {
        assert!(self.context.is_some() && self.renderer.is_some());
        self.context.as_mut().unwrap().begin_frame();
        // Render UI
        if self.layout.is_some() {
            (self.layout.unwrap())();
        }
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

    /// Registers a user image to be used by egui
    pub fn register_user_image(
        &mut self,
        image: Arc<dyn ImageViewAccess + Send + Sync>,
    ) -> egui::TextureId {
        assert!(self.context.is_some() && self.renderer.is_some());
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
