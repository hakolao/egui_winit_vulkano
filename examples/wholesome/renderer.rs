use std::sync::Arc;

use cgmath::{Matrix4, SquareMatrix};
use egui_winit_vulkano::Gui;
use vulkano::{
    device::{Device, DeviceExtensions, Features, Queue},
    framebuffer::{RenderPassAbstract, Subpass},
    image::{AttachmentImage, ImageAccess, ImageUsage, SwapchainImage},
    instance::{Instance, InstanceExtensions, PhysicalDevice},
    swapchain,
    swapchain::{
        AcquireError, ColorSpace, FullscreenExclusive, PresentMode, Surface, SurfaceTransform,
        Swapchain, SwapchainCreationError,
    },
    sync,
    sync::{FlushError, GpuFuture},
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
    color_images: Vec<Arc<SwapchainImage<Window>>>,
    scene_images: Vec<Arc<AttachmentImage>>,
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
        width: u32,
        height: u32,
        scene_view_size: [u32; 2],
        present_mode: PresentMode,
        name: &str,
    ) -> Self {
        // Add instance extensions based on needs
        let instance_extensions = InstanceExtensions { ..vulkano_win::required_extensions() };
        // Create instance
        let instance =
            Instance::new(None, &instance_extensions, None).expect("Failed to create instance");
        // Get most performant device (physical)
        let physical = PhysicalDevice::enumerate(&instance)
            .fold(None, |acc, val| {
                if acc.is_none() {
                    Some(val)
                } else {
                    if acc.unwrap().limits().max_compute_shared_memory_size()
                        >= val.limits().max_compute_shared_memory_size()
                    {
                        acc
                    } else {
                        Some(val)
                    }
                }
            })
            .expect("No physical device found");
        println!("Using device: {} (type: {:?})", physical.name(), physical.ty());
        // Create rendering surface along with window
        let surface = WindowBuilder::new()
            .with_inner_size(winit::dpi::LogicalSize::new(width, height))
            .with_title(name)
            .build_vk_surface(&event_loop, instance.clone())
            .expect("Failed to create vulkan surface & window");
        // Create device
        let (device, queue) = Self::create_device(physical, surface.clone());
        // Create swap chain & frame(s) to which we'll render
        let (swap_chain, images) = Self::create_swap_chain(
            surface.clone(),
            physical,
            device.clone(),
            queue.clone(),
            present_mode,
        );
        let previous_frame_end = Some(sync::now(device.clone()).boxed());
        // Create frame system
        let render_pass_system = FrameSystem::new(queue.clone(), swap_chain.format());
        let scene = TriangleDrawSystem::new(queue.clone(), render_pass_system.deferred_subpass());

        let scene_images = Self::create_scene_images(device.clone(), &images, scene_view_size);
        Self {
            instance,
            device,
            surface,
            queue,
            swap_chain,
            color_images: images,
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
    /// We need khr_external_memory_fd for CUDA + Vulkan interoperability
    fn create_device(
        physical: PhysicalDevice,
        surface: Arc<Surface<Window>>,
    ) -> (Arc<Device>, Arc<Queue>) {
        let queue_family = physical
            .queue_families()
            .find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false))
            .expect("couldn't find a graphical queue family");
        // Add device extensions based on needs
        let device_extensions =
            DeviceExtensions { ..DeviceExtensions::supported_by_device(physical) };
        // Add device features
        let features = Features { ..*physical.supported_features() };
        let (device, mut queues) = {
            Device::new(
                physical,
                &features,
                &device_extensions,
                [(queue_family, 0.5)].iter().cloned(),
            )
            .expect("failed to create device")
        };
        (device, queues.next().unwrap())
    }

    fn create_swap_chain(
        surface: Arc<Surface<Window>>,
        physical: PhysicalDevice,
        device: Arc<Device>,
        queue: Arc<Queue>,
        present_mode: PresentMode,
    ) -> (Arc<Swapchain<Window>>, Vec<Arc<SwapchainImage<Window>>>) {
        let (swap_chain, images) = {
            let caps = surface.capabilities(physical).unwrap();
            let alpha = caps.supported_composite_alpha.iter().next().unwrap();
            let format = caps.supported_formats[0].0;
            let dimensions: [u32; 2] = surface.window().inner_size().into();
            Swapchain::new(
                device.clone(),
                surface.clone(),
                caps.min_image_count,
                format,
                dimensions,
                1,
                ImageUsage::color_attachment(),
                &queue,
                SurfaceTransform::Identity,
                alpha,
                present_mode,
                FullscreenExclusive::Default,
                true,
                ColorSpace::SrgbNonLinear,
            )
            .unwrap()
        };
        (swap_chain, images)
    }

    fn create_scene_images(
        device: Arc<Device>,
        swapchain_images: &Vec<Arc<SwapchainImage<Window>>>,
        scene_view_size: [u32; 2],
    ) -> Vec<Arc<AttachmentImage>> {
        let mut scene_images = vec![];
        for si in swapchain_images {
            let image = AttachmentImage::sampled_input_attachment(
                device.clone(),
                scene_view_size,
                ImageAccess::format(&si),
            )
            .expect("Failed to create scene image");
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

    // Return a deferred subpass for our render pass
    pub fn deferred_subpass(&self) -> Subpass<Arc<dyn RenderPassAbstract + Send + Sync>> {
        self.frame_system.deferred_subpass()
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

    pub fn scene_images(&mut self) -> &Vec<Arc<AttachmentImage>> {
        &self.scene_images
    }

    pub fn render(&mut self, gui: &mut Gui) {
        // Recreate swap chain if needed (when resizing of window occurs or swapchain is outdated)
        if self.recreate_swapchain {
            self.recreate_swapchain();
        }
        // Acquire next image in the swapchain
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
        // Acquire frame to which we'll render (by image_num)
        let future = self.previous_frame_end.take().unwrap().join(acquire_future);
        let mut frame = self.frame_system.frame(
            future,
            self.color_images[image_num].clone(),
            Matrix4::identity(),
        );
        // Draw each render pass
        let mut after_future = None;
        while let Some(pass) = frame.next_pass() {
            match pass {
                Pass::Deferred(mut draw_pass) => {
                    // Render UI
                    let cb = gui.draw(self.surface.window(), draw_pass.viewport_dimensions());
                    draw_pass.execute(cb);
                }
                Pass::Finished(af) => {
                    after_future = Some(af);
                }
            }
        }
        // Finish render
        self.finish(after_future, image_num);
    }

    /// Renders the pass for scene on scene images
    fn render_scene(&mut self) {
        let future = sync::now(self.device.clone()).boxed();
        let mut frame = self.frame_system.frame(
            future,
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
            Self::create_scene_images(self.device.clone(), &self.color_images, new_size);
        self.scene_images = scene_images;
    }

    /// Swapchain is recreated when resized
    fn recreate_swapchain(&mut self) {
        let dimensions: [u32; 2] = self.surface.window().inner_size().into();
        let (new_swapchain, new_images) = match self.swap_chain.recreate_with_dimensions(dimensions)
        {
            Ok(r) => r,
            Err(SwapchainCreationError::UnsupportedDimensions) => return,
            Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
        };
        self.swap_chain = new_swapchain;
        self.color_images = new_images;
        self.recreate_swapchain = false;
    }

    /// Finishes render by presenting the swapchain
    fn finish(&mut self, after_future: Option<Box<dyn GpuFuture>>, image_num: usize) {
        let future = after_future
            .unwrap()
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
