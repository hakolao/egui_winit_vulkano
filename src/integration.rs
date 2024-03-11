// Copyright (c) 2021 Okko Hakola
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.
use std::sync::Arc;

use egui::{ClippedPrimitive, TexturesDelta};
use egui_winit::winit::event_loop::EventLoopWindowTarget;
use vulkano::{
    command_buffer::SecondaryAutoCommandBuffer,
    device::Queue,
    format::{Format, NumericFormat},
    image::{sampler::SamplerCreateInfo, view::ImageView, SampleCount},
    render_pass::Subpass,
    swapchain::Surface,
    sync::GpuFuture,
};
use winit::window::Window;

#[cfg(feature = "image")]
use crate::utils::immutable_texture_from_file;
use crate::{
    renderer::{RenderResources, Renderer},
    utils::immutable_texture_from_bytes,
};

pub struct GuiConfig {
    /// Allows supplying sRGB ImageViews as render targets instead of just UNORM ImageViews, defaults to false.
    /// **Using sRGB will cause minor discoloration of UI elements** due to blending in linear color space and not
    /// sRGB as Egui expects.
    ///
    /// If you would like to visually compare between UNORM and sRGB render targets, run the `demo_app` example of
    /// this crate.
    pub allow_srgb_render_target: bool,
    /// Whether to render gui as overlay. Only relevant in the case of `Gui::new`, not when using
    /// subpass. Determines whether the pipeline should clear the target image.
    pub is_overlay: bool,
    /// Multisample count. Defaults to 1. If you use more than 1, you'll have to ensure your
    /// pipeline and target image matches that.
    pub samples: SampleCount,
}

impl Default for GuiConfig {
    fn default() -> Self {
        GuiConfig {
            allow_srgb_render_target: false,
            is_overlay: false,
            samples: SampleCount::Sample1,
        }
    }
}

impl GuiConfig {
    pub fn validate(&self, output_format: Format) {
        if output_format.numeric_format_color().unwrap() == NumericFormat::SRGB {
            assert!(
                self.allow_srgb_render_target,
                "Using an output format with sRGB requires `GuiConfig::allow_srgb_render_target` \
                 to be set! Egui prefers UNORM render targets. Using sRGB will cause minor \
                 discoloration of UI elements due to blending in linear color space and not sRGB \
                 as Egui expects."
            );
        }
    }
}

pub struct Gui {
    pub egui_ctx: egui::Context,
    pub egui_winit: egui_winit::State,
    renderer: Renderer,
    surface: Arc<Surface>,

    shapes: Vec<egui::epaint::ClippedShape>,
    textures_delta: egui::TexturesDelta,
}

impl Gui {
    /// Creates new Egui to Vulkano integration by setting the necessary parameters
    /// This is to be called once we have access to vulkano_win's winit window surface
    /// and gfx queue. Created with this, the renderer will own a render pass which is useful to e.g. place your render pass' images
    /// onto egui windows
    pub fn new<T>(
        event_loop: &EventLoopWindowTarget<T>,
        surface: Arc<Surface>,
        gfx_queue: Arc<Queue>,
        output_format: Format,
        config: GuiConfig,
    ) -> Gui {
        config.validate(output_format);
        let renderer = Renderer::new_with_render_pass(
            gfx_queue,
            output_format,
            config.is_overlay,
            config.samples,
        );
        Self::new_internal(event_loop, surface, renderer)
    }

    /// Same as `new` but instead of integration owning a render pass, egui renders on your subpass
    pub fn new_with_subpass<T>(
        event_loop: &EventLoopWindowTarget<T>,
        surface: Arc<Surface>,
        gfx_queue: Arc<Queue>,
        subpass: Subpass,
        output_format: Format,
        config: GuiConfig,
    ) -> Gui {
        config.validate(output_format);
        let renderer = Renderer::new_with_subpass(gfx_queue, output_format, subpass);
        Self::new_internal(event_loop, surface, renderer)
    }

    /// Same as `new` but instead of integration owning a render pass, egui renders on your subpass
    fn new_internal<T>(
        event_loop: &EventLoopWindowTarget<T>,
        surface: Arc<Surface>,
        renderer: Renderer,
    ) -> Gui {
        let max_texture_side =
            renderer.queue().device().physical_device().properties().max_image_dimension2_d
                as usize;
        let egui_ctx: egui::Context = Default::default();
        let egui_winit = egui_winit::State::new(
            egui_ctx.viewport_id(),
            event_loop,
            Some(surface_window(&surface).scale_factor() as f32),
            Some(max_texture_side),
        );
        Gui {
            egui_ctx,
            egui_winit,
            renderer,
            surface,
            shapes: vec![],
            textures_delta: Default::default(),
        }
    }

    /// Returns the pixels per point of the window of this gui.
    fn pixels_per_point(&self) -> f32 {
        egui_winit::pixels_per_point(&self.egui_ctx, surface_window(&self.surface))
    }

    /// Returns a set of resources used to construct the render pipeline. These can be reused
    /// to create additional pipelines and buffers to be rendered in a `PaintCallback`.
    pub fn render_resources(&self) -> RenderResources {
        self.renderer.render_resources()
    }

    /// Updates context state by winit window event.
    /// Returns `true` if egui wants exclusive use of this event
    /// (e.g. a mouse click on an egui window, or entering text into a text field).
    /// For instance, if you use egui for a game, you want to first call this
    /// and only when this returns `false` pass on the events to your game.
    ///
    /// Note that egui uses `tab` to move focus between elements, so this will always return `true` for tabs.
    pub fn update(&mut self, winit_event: &winit::event::WindowEvent<'_>) -> bool {
        self.egui_winit.on_window_event(&self.egui_ctx, winit_event).consumed
    }

    /// Begins Egui frame & determines what will be drawn later. This must be called before draw, and after `update` (winit event).
    pub fn immediate_ui(&mut self, layout_function: impl FnOnce(&mut Self)) {
        let raw_input = self.egui_winit.take_egui_input(surface_window(&self.surface));
        self.egui_ctx.begin_frame(raw_input);
        // Render Egui
        layout_function(self);
    }

    /// If you wish to better control when to begin frame, do so by calling this function
    /// (Finish by drawing)
    pub fn begin_frame(&mut self) {
        let raw_input = self.egui_winit.take_egui_input(surface_window(&self.surface));
        self.egui_ctx.begin_frame(raw_input);
    }

    /// Renders ui on `final_image` & Updates cursor icon
    /// Finishes Egui frame
    /// - `before_future` = Vulkano's GpuFuture
    /// - `final_image` = Vulkano's image (render target)
    pub fn draw_on_image<F>(
        &mut self,
        before_future: F,
        final_image: Arc<ImageView>,
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

        let (clipped_meshes, textures_delta) = self.extract_draw_data_at_frame_end();

        self.renderer.draw_on_image(
            &clipped_meshes,
            &textures_delta,
            self.pixels_per_point(),
            before_future,
            final_image,
        )
    }

    /// Creates commands for rendering ui on subpass' image and returns the command buffer for execution on your side
    /// - Finishes Egui frame
    /// - You must execute the secondary command buffer yourself
    pub fn draw_on_subpass_image(
        &mut self,
        image_dimensions: [u32; 2],
    ) -> Arc<SecondaryAutoCommandBuffer> {
        if self.renderer.has_renderpass() {
            panic!(
                "Gui integration has been created with its own render pass, use `draw_on_image` \
                 instead"
            )
        }

        let (clipped_meshes, textures_delta) = self.extract_draw_data_at_frame_end();

        self.renderer.draw_on_subpass_image(
            &clipped_meshes,
            &textures_delta,
            self.pixels_per_point(),
            image_dimensions,
        )
    }

    fn extract_draw_data_at_frame_end(&mut self) -> (Vec<ClippedPrimitive>, TexturesDelta) {
        self.end_frame();
        let shapes = std::mem::take(&mut self.shapes);
        let textures_delta = std::mem::take(&mut self.textures_delta);
        let clipped_meshes = self.egui_ctx.tessellate(shapes, self.pixels_per_point());
        (clipped_meshes, textures_delta)
    }

    fn end_frame(&mut self) {
        let egui::FullOutput {
            platform_output,
            textures_delta,
            shapes,
            pixels_per_point: _,
            viewport_output: _,
        } = self.egui_ctx.end_frame();

        self.egui_winit.handle_platform_output(
            surface_window(&self.surface),
            &self.egui_ctx,
            platform_output,
        );
        self.shapes = shapes;
        self.textures_delta = textures_delta;
    }

    /// Registers a user image from Vulkano image view to be used by egui
    pub fn register_user_image_view(
        &mut self,
        image: Arc<ImageView>,
        sampler_create_info: SamplerCreateInfo,
    ) -> egui::TextureId {
        self.renderer.register_image(image, sampler_create_info)
    }

    /// Registers a user image to be used by egui
    /// - `image_file_bytes`: e.g. include_bytes!("./assets/tree.png")
    /// - `format`: e.g. vulkano::format::Format::R8G8B8A8Unorm
    #[cfg(feature = "image")]
    pub fn register_user_image(
        &mut self,
        image_file_bytes: &[u8],
        format: vulkano::format::Format,
        sampler_create_info: SamplerCreateInfo,
    ) -> egui::TextureId {
        let image = immutable_texture_from_file(
            self.renderer.allocators(),
            self.renderer.queue(),
            image_file_bytes,
            format,
        )
        .expect("Failed to create image");
        self.renderer.register_image(image, sampler_create_info)
    }

    pub fn register_user_image_from_bytes(
        &mut self,
        image_byte_data: &[u8],
        dimensions: [u32; 2],
        format: vulkano::format::Format,
        sampler_create_info: SamplerCreateInfo,
    ) -> egui::TextureId {
        let image = immutable_texture_from_bytes(
            self.renderer.allocators(),
            self.renderer.queue(),
            image_byte_data,
            dimensions,
            format,
        )
        .expect("Failed to create image");
        self.renderer.register_image(image, sampler_create_info)
    }

    /// Unregisters a user image
    pub fn unregister_user_image(&mut self, texture_id: egui::TextureId) {
        self.renderer.unregister_image(texture_id);
    }

    /// Access egui's context (which can be used to e.g. set fonts, visuals etc)
    pub fn context(&self) -> egui::Context {
        self.egui_ctx.clone()
    }
}

// Helper to retrieve Window from surface object
fn surface_window(surface: &Surface) -> &Window {
    surface.object().unwrap().downcast_ref::<Window>().unwrap()
}
