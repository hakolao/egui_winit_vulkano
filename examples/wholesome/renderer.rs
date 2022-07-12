// Copyright (c) 2021 Okko Hakola
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;

use cgmath::{Matrix4, SquareMatrix};
use vulkano::{device::Queue, image::ImageViewAbstract, sync::GpuFuture};
use vulkano_util::renderer::DeviceImageView;

use crate::{
    frame_system::{FrameSystem, Pass},
    triangle_draw_system::TriangleDrawSystem,
};

pub struct RenderPipeline {
    frame_system: FrameSystem,
    draw_pipeline: TriangleDrawSystem,
}

impl RenderPipeline {
    pub fn new(queue: Arc<Queue>, image_format: vulkano::format::Format) -> Self {
        let frame_system = FrameSystem::new(queue.clone(), image_format);
        let draw_pipeline = TriangleDrawSystem::new(queue.clone(), frame_system.deferred_subpass());

        Self { frame_system, draw_pipeline }
    }

    /// Renders the pass for scene on scene images
    pub fn render(
        &mut self,
        before_future: Box<dyn GpuFuture>,
        image: DeviceImageView,
    ) -> Box<dyn GpuFuture> {
        let mut frame = self.frame_system.frame(
            before_future,
            // Notice that final image is now scene image
            image.clone(),
            Matrix4::identity(),
        );
        let dims = image.dimensions().width_height();
        // Draw each render pass that's related to scene
        let mut after_future = None;
        while let Some(pass) = frame.next_pass() {
            match pass {
                Pass::Deferred(mut draw_pass) => {
                    let cb = self.draw_pipeline.draw(dims);
                    draw_pass.execute(cb);
                }
                Pass::Finished(af) => {
                    after_future = Some(af);
                }
            }
        }
        after_future.unwrap().then_signal_fence_and_flush().unwrap().boxed()
    }
}
