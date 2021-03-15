// Copyright (c) 2017 The vulkano developers <=== !
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

// This is a simplified version of the example. See that for commented version of this code.
// https://github.com/vulkano-rs/vulkano-examples/blob/master/src/bin/deferred/frame/system.rs
// Egui drawing could be its own pass or it could be a deferred subpass

use std::sync::Arc;

use cgmath::Matrix4;
use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, CommandBuffer, SubpassContents},
    device::Queue,
    format::Format,
    framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass},
    image::{view::ImageView, AttachmentImage, ImageViewAbstract},
    sync::GpuFuture,
};

/// System that contains the necessary facilities for rendering a single frame.
pub struct FrameSystem {
    gfx_queue: Arc<Queue>,
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    depth_buffer: Arc<ImageView<Arc<AttachmentImage>>>,
}

impl FrameSystem {
    pub fn new(gfx_queue: Arc<Queue>, final_output_format: Format) -> FrameSystem {
        let render_pass = Arc::new(
            vulkano::ordered_passes_renderpass!(gfx_queue.device().clone(),
                attachments: {
                    final_color: {
                        load: Clear,
                        store: Store,
                        format: final_output_format,
                        samples: 1,
                    },
                    depth: {
                        load: Clear,
                        store: DontCare,
                        format: Format::D16Unorm,
                        samples: 1,
                    }
                },
                passes: [
                    {
                        color: [final_color],
                        depth_stencil: {depth},
                        input: []
                    }
                ]
            )
            .unwrap(),
        );
        let depth_buffer = ImageView::new(
            AttachmentImage::transient_input_attachment(
                gfx_queue.device().clone(),
                [1, 1],
                Format::D16Unorm,
            )
            .unwrap(),
        )
        .unwrap();
        FrameSystem { gfx_queue, render_pass: render_pass as Arc<_>, depth_buffer }
    }

    #[inline]
    pub fn deferred_subpass(&self) -> Subpass<Arc<dyn RenderPassAbstract + Send + Sync>> {
        Subpass::from(self.render_pass.clone(), 0).unwrap()
    }

    pub fn frame<F, I>(
        &mut self,
        before_future: F,
        final_image: I,
        world_to_framebuffer: Matrix4<f32>,
    ) -> Frame
    where
        F: GpuFuture + 'static,
        I: ImageViewAbstract + Clone + Send + Sync + 'static,
    {
        let img_dims = final_image.image().dimensions().width_height();
        if self.depth_buffer.image().dimensions().width_height() != img_dims {
            self.depth_buffer = ImageView::new(
                AttachmentImage::transient_input_attachment(
                    self.gfx_queue.device().clone(),
                    img_dims,
                    Format::D16Unorm,
                )
                .unwrap(),
            )
            .unwrap();
        }
        let framebuffer = Arc::new(
            Framebuffer::start(self.render_pass.clone())
                .add(final_image.clone())
                .unwrap()
                .add(self.depth_buffer.clone())
                .unwrap()
                .build()
                .unwrap(),
        );
        let mut command_buffer_builder = AutoCommandBufferBuilder::primary_one_time_submit(
            self.gfx_queue.device().clone(),
            self.gfx_queue.family(),
        )
        .unwrap();
        command_buffer_builder
            .begin_render_pass(framebuffer.clone(), SubpassContents::SecondaryCommandBuffers, vec![
                [0.0, 0.0, 0.0, 0.0].into(),
                1.0f32.into(),
            ])
            .unwrap();

        Frame {
            system: self,
            before_main_cb_future: Some(Box::new(before_future)),
            framebuffer,
            num_pass: 0,
            command_buffer_builder: Some(command_buffer_builder),
            world_to_framebuffer,
        }
    }
}

pub struct Frame<'a> {
    system: &'a mut FrameSystem,
    num_pass: u8,
    before_main_cb_future: Option<Box<dyn GpuFuture>>,
    framebuffer: Arc<dyn FramebufferAbstract + Send + Sync>,
    command_buffer_builder: Option<AutoCommandBufferBuilder>,
    #[allow(dead_code)]
    world_to_framebuffer: Matrix4<f32>,
}

impl<'a> Frame<'a> {
    pub fn next_pass<'f>(&'f mut self) -> Option<Pass<'f, 'a>> {
        match {
            let current_pass = self.num_pass;
            self.num_pass += 1;
            current_pass
        } {
            0 => Some(Pass::Deferred(DrawPass { frame: self })),
            1 => {
                self.command_buffer_builder.as_mut().unwrap().end_render_pass().unwrap();
                let command_buffer = self.command_buffer_builder.take().unwrap().build().unwrap();
                let after_main_cb = self
                    .before_main_cb_future
                    .take()
                    .unwrap()
                    .then_execute(self.system.gfx_queue.clone(), command_buffer)
                    .unwrap();
                Some(Pass::Finished(Box::new(after_main_cb)))
            }
            _ => None,
        }
    }
}

pub enum Pass<'f, 's: 'f> {
    Deferred(DrawPass<'f, 's>),
    Finished(Box<dyn GpuFuture>),
}

pub struct DrawPass<'f, 's: 'f> {
    frame: &'f mut Frame<'s>,
}

impl<'f, 's: 'f> DrawPass<'f, 's> {
    #[inline]
    pub fn execute<C>(&mut self, command_buffer: C)
    where
        C: CommandBuffer + Send + Sync + 'static,
    {
        unsafe {
            self.frame
                .command_buffer_builder
                .as_mut()
                .unwrap()
                .execute_commands(command_buffer)
                .unwrap();
        }
    }

    #[allow(dead_code)]
    #[inline]
    pub fn viewport_dimensions(&self) -> [u32; 2] {
        let dims = self.frame.framebuffer.dimensions();
        [dims[0], dims[1]]
    }

    #[allow(dead_code)]
    #[inline]
    pub fn world_to_framebuffer_matrix(&self) -> Matrix4<f32> {
        self.frame.world_to_framebuffer
    }
}
