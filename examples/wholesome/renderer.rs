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
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::{
    command_buffer::allocator::StandardCommandBufferAllocator, device::Queue,
    memory::allocator::StandardMemoryAllocator, sync::GpuFuture,
};

use crate::{
    frame_system::{FrameSystem, Pass},
    triangle_draw_system::TriangleDrawSystem,
};

pub struct RenderPipeline {
    frame_system: FrameSystem,
    draw_pipeline: TriangleDrawSystem,
}

#[derive(Clone)]
pub struct Allocators {
    pub command_buffers: Arc<StandardCommandBufferAllocator>,
    pub memory: Arc<StandardMemoryAllocator>,
}

impl RenderPipeline {
    pub fn new(queue: Arc<Queue>, image_format: Format, allocators: &Allocators) -> Self {
        let frame_system = FrameSystem::new(queue.clone(), image_format, allocators.clone());
        let draw_pipeline =
            TriangleDrawSystem::new(queue, frame_system.deferred_subpass(), allocators);

        Self { frame_system, draw_pipeline }
    }

    /// Renders the pass for scene on scene images
    pub fn render(
        &mut self,
        before_future: Box<dyn GpuFuture>,
        image: Arc<ImageView>,
    ) -> Box<dyn GpuFuture> {
        let mut frame = self.frame_system.frame(
            before_future,
            // Notice that final image is now scene image
            image.clone(),
            Matrix4::identity(),
        );
        let dims = image.image().extent();
        // Draw each render pass that's related to scene
        let mut after_future = None;
        while let Some(pass) = frame.next_pass() {
            match pass {
                Pass::Deferred(mut draw_pass) => {
                    let cb = Arc::new(self.draw_pipeline.draw([dims[0], dims[1]]));
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
