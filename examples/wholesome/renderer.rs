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
use egui_winit_vulkano::Gui;
use vulkano::{
    device::{
        physical::PhysicalDevice, Device, DeviceCreateInfo, DeviceExtensions, Features, Queue,
        QueueCreateInfo,
    },
    image::{view::ImageView, AttachmentImage, ImageUsage, ImageViewAbstract, SwapchainImage},
    instance::{Instance, InstanceCreateInfo, InstanceExtensions},
    swapchain,
    swapchain::{
        AcquireError, PresentMode, Surface, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
    },
    sync,
    sync::{FlushError, GpuFuture},
    Version,
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};

use crate::{
    frame_system::{FrameSystem, Pass},
    triangle_draw_system::TriangleDrawSystem,
};

pub struct Renderer {
    #[allow(dead_code)]
    instance: Arc<Instance>,
    device: Arc<Device>,
    surface: Arc<Surface<Window>>,
    queue: Arc<Queue>,
    swap_chain: Arc<Swapchain<Window>>,
    final_images: Vec<Arc<ImageView<SwapchainImage<Window>>>>,
    scene_images: Vec<Arc<ImageView<AttachmentImage>>>,
    image_num: usize,
    scene_view_size: [u32; 2],
    recreate_swapchain: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
    frame_system: FrameSystem,
    scene: TriangleDrawSystem,
}

impl Renderer {
    pub fn new(
        event_loop: &EventLoop<()>,
        window_size: [u32; 2],
        scene_view_size: [u32; 2],
        present_mode: PresentMode,
        name: &str,
    ) -> Self {
        // Add instance extensions based on needs
        let instance_extensions = InstanceExtensions { ..vulkano_win::required_extensions() };
        // Create instance
        let instance = Instance::new(InstanceCreateInfo {
            application_version: Version::V1_2,
            enabled_extensions: instance_extensions,
            ..Default::default()
        })
        .expect("Failed to create instance");
        // Get most performant device (physical)
        let physical = PhysicalDevice::enumerate(&instance)
            .fold(None, |acc, val| {
                if acc.is_none() {
                    Some(val)
                } else if acc.unwrap().properties().max_compute_shared_memory_size
                    >= val.properties().max_compute_shared_memory_size
                {
                    acc
                } else {
                    Some(val)
                }
            })
            .expect("No physical device found");
        println!("Using device {}", physical.properties().device_name);
        // Create rendering surface along with window
        let surface = WindowBuilder::new()
            .with_inner_size(winit::dpi::LogicalSize::new(window_size[0], window_size[1]))
            .with_title(name)
            .build_vk_surface(event_loop, instance.clone())
            .expect("Failed to create vulkan surface & window");
        // Create device
        let (device, queue) = Self::create_device(physical, surface.clone());
        // Create swap chain & frame(s) to which we'll render
        let (swap_chain, images) =
            Self::create_swap_chain(surface.clone(), physical, device.clone(), present_mode);
        let previous_frame_end = Some(sync::now(device.clone()).boxed());
        // Create frame system
        let render_pass_system = FrameSystem::new(queue.clone(), images[0].format().unwrap());
        let scene = TriangleDrawSystem::new(queue.clone(), render_pass_system.deferred_subpass());

        let scene_images = Self::create_scene_images(device.clone(), &images, scene_view_size);
        Self {
            instance,
            device,
            surface,
            queue,
            swap_chain,
            final_images: images,
            scene_images,
            image_num: 0,
            scene_view_size,
            previous_frame_end,
            recreate_swapchain: false,
            frame_system: render_pass_system,
            scene,
        }
    }

    /// Creates vulkan device with required queue families and required extensions
    fn create_device(
        physical: PhysicalDevice,
        surface: Arc<Surface<Window>>,
    ) -> (Arc<Device>, Arc<Queue>) {
        let queue_family = physical
            .queue_families()
            .find(|&q| q.supports_graphics() && q.supports_surface(&surface).unwrap_or(false))
            .expect("couldn't find a graphical queue family");
        // Add device extensions based on needs
        let device_extensions =
            DeviceExtensions { khr_swapchain: true, ..DeviceExtensions::none() };
        // Add device features
        let features = Features::none();
        let (device, mut queues) = {
            Device::new(physical, DeviceCreateInfo {
                enabled_extensions: physical.required_extensions().union(&device_extensions),
                enabled_features: features,
                queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
                _ne: Default::default(),
            })
            .expect("failed to create device")
        };
        (device, queues.next().unwrap())
    }

    fn create_swap_chain(
        surface: Arc<Surface<Window>>,
        physical: PhysicalDevice,
        device: Arc<Device>,
        present_mode: PresentMode,
    ) -> (Arc<Swapchain<Window>>, Vec<Arc<ImageView<SwapchainImage<Window>>>>) {
        let surface_capabilities =
            physical.surface_capabilities(&surface, Default::default()).unwrap();
        let image_format =
            Some(physical.surface_formats(&surface, Default::default()).unwrap()[0].0);
        let image_extent = surface.window().inner_size().into();

        let (swapchain, images) = Swapchain::new(device, surface, SwapchainCreateInfo {
            min_image_count: surface_capabilities.min_image_count,
            image_format,
            image_extent,
            image_usage: ImageUsage::color_attachment(),
            composite_alpha: surface_capabilities.supported_composite_alpha.iter().next().unwrap(),
            present_mode,
            ..Default::default()
        })
        .unwrap();
        let images = images
            .into_iter()
            .map(|image| ImageView::new_default(image).unwrap())
            .collect::<Vec<_>>();
        (swapchain, images)
    }

    fn create_scene_images(
        device: Arc<Device>,
        swapchain_images: &Vec<Arc<ImageView<SwapchainImage<Window>>>>,
        scene_view_size: [u32; 2],
    ) -> Vec<Arc<ImageView<AttachmentImage>>> {
        let mut scene_images = vec![];
        for si in swapchain_images {
            let image = ImageView::new_default(
                AttachmentImage::sampled_input_attachment(
                    device.clone(),
                    scene_view_size,
                    si.format().unwrap(),
                )
                .expect("Failed to create scene image"),
            )
            .unwrap();
            scene_images.push(image);
        }
        scene_images
    }

    #[allow(dead_code)]
    pub fn device(&self) -> Arc<Device> {
        self.device.clone()
    }

    pub fn queue(&self) -> Arc<Queue> {
        self.queue.clone()
    }

    pub fn surface(&self) -> Arc<Surface<Window>> {
        self.surface.clone()
    }

    pub fn window(&self) -> &Window {
        self.surface.window()
    }

    pub fn resize(&mut self) {
        self.recreate_swapchain = true;
    }

    pub fn last_image_num(&self) -> usize {
        self.image_num
    }

    pub fn scene_images(&mut self) -> &Vec<Arc<ImageView<AttachmentImage>>> {
        &self.scene_images
    }

    /// Renders scene onto scene images using frame system and finally draws UI on final
    /// swapchain images
    pub fn render(&mut self, gui: &mut Gui) {
        // Recreate swap chain if needed (when resizing of window occurs or swapchain is outdated)
        if self.recreate_swapchain {
            self.recreate_swapchain();
        }
        // Acquire next image in the swapchain and our image num index
        let (image_num, suboptimal, acquire_future) =
            match swapchain::acquire_next_image(self.swap_chain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    self.recreate_swapchain = true;
                    return;
                }
                Err(e) => panic!("Failed to acquire next image: {:?}", e),
            };
        if suboptimal {
            self.recreate_swapchain = true;
        }
        self.image_num = image_num;
        // Knowing image num, let's render our scene on scene images
        self.render_scene();
        // Finally render GUI on our swapchain color image attachments
        let future = self.previous_frame_end.take().unwrap().join(acquire_future);
        let after_future = gui.draw_on_image(future, self.final_images[image_num].clone());
        // Finish render
        self.finish(after_future, image_num);
    }

    /// Renders the pass for scene on scene images
    fn render_scene(&mut self) {
        let future = sync::now(self.device.clone()).boxed();
        // Acquire frame from our frame system
        let mut frame = self.frame_system.frame(
            future,
            // Notice that final image is now scene image
            self.scene_images[self.image_num].clone(),
            Matrix4::identity(),
        );
        // Draw each render pass that's related to scene
        let mut after_future = None;
        while let Some(pass) = frame.next_pass() {
            match pass {
                Pass::Deferred(mut draw_pass) => {
                    let cb = self.scene.draw(self.scene_view_size);
                    draw_pass.execute(cb);
                }
                Pass::Finished(af) => {
                    after_future = Some(af);
                }
            }
        }
        // Wait on our future
        let future = after_future
            .unwrap()
            .then_signal_fence_and_flush()
            .expect("Failed to signal fence and flush");
        match future.wait(None) {
            Ok(x) => x,
            Err(err) => println!("err: {:?}", err),
        }
    }

    #[allow(dead_code)]
    fn resize_scene_view(&mut self, new_size: [u32; 2]) {
        self.scene_view_size = new_size;
        let scene_images =
            Self::create_scene_images(self.device.clone(), &self.final_images, new_size);
        self.scene_images = scene_images;
    }

    /// Swapchain is recreated when resized
    fn recreate_swapchain(&mut self) {
        let dimensions: [u32; 2] = self.surface.window().inner_size().into();
        let (new_swapchain, new_images) = match self.swap_chain.recreate(SwapchainCreateInfo {
            image_extent: dimensions,
            ..self.swap_chain.create_info()
        }) {
            Ok(r) => r,
            Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
            Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
        };
        self.swap_chain = new_swapchain;
        let new_images = new_images
            .into_iter()
            .map(|image| ImageView::new_default(image).unwrap())
            .collect::<Vec<_>>();
        self.final_images = new_images;

        self.recreate_swapchain = false;
    }

    /// Finishes render by presenting the swapchain
    fn finish(&mut self, after_future: Box<dyn GpuFuture>, image_num: usize) {
        let future = after_future
            .then_swapchain_present(self.queue.clone(), self.swap_chain.clone(), image_num)
            .then_signal_fence_and_flush();
        match future {
            Ok(future) => {
                // A hack to prevent OutOfMemory error on Nvidia :(
                // https://github.com/vulkano-rs/vulkano/issues/627
                match future.wait(None) {
                    Ok(x) => x,
                    Err(err) => println!("err: {:?}", err),
                }
                self.previous_frame_end = Some(future.boxed());
            }
            Err(FlushError::OutOfDate) => {
                self.recreate_swapchain = true;
                self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
            }
            Err(e) => {
                println!("Failed to flush future: {:?}", e);
                self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
            }
        }
    }
}
