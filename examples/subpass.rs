// Copyright (c) 2021 Okko Hakola
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::{convert::TryFrom, sync::Arc};

use bytemuck::{Pod, Zeroable};
use egui::{ScrollArea, TextEdit, TextStyle};
use egui_winit_vulkano::Gui;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SubpassContents},
    device::{
        physical::PhysicalDevice, Device, DeviceCreateInfo, DeviceExtensions, Features, Queue,
        QueueCreateInfo,
    },
    format::Format,
    image::{view::ImageView, ImageAccess, ImageUsage, ImageViewAbstract, SwapchainImage},
    instance::{Instance, InstanceCreateInfo, InstanceExtensions},
    pipeline::{
        graphics::{
            input_assembly::InputAssemblyState,
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
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
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

pub fn main() {
    // Winit event loop & our time tracking initialization
    let event_loop = EventLoop::new();
    // Create renderer for our scene & ui
    let window_size = [1280, 720];
    let mut renderer =
        SimpleGuiRenderer::new(&event_loop, window_size, PresentMode::Fifo, "Minimal");
    // After creating the renderer (window, gfx_queue) create out gui integration using gui subpass from renderer
    let mut gui = Gui::new_with_subpass(renderer.surface(), renderer.queue(), renderer.gui_pass());
    // Create gui state (pass anything your state requires)
    let mut code = CODE.to_owned();
    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent { event, window_id }
                if window_id == renderer.surface().window().id() =>
            {
                // Update Egui integration so the UI works!
                let _pass_events_to_game = !gui.update(&event);
                match event {
                    WindowEvent::Resized(_) => {
                        renderer.resize();
                    }
                    WindowEvent::ScaleFactorChanged { .. } => {
                        renderer.resize();
                    }
                    WindowEvent::CloseRequested => {
                        *control_flow = ControlFlow::Exit;
                    }
                    _ => (),
                }
            }
            Event::RedrawRequested(window_id) if window_id == window_id => {
                // Set immediate UI in redraw here
                gui.immediate_ui(|gui| {
                    let ctx = gui.context();
                    egui::CentralPanel::default().show(&ctx, |ui| {
                        ui.vertical_centered(|ui| {
                            ui.add(egui::widgets::Label::new("Hi there!"));
                        });
                        ui.separator();
                        ui.columns(2, |columns| {
                            ScrollArea::vertical().id_source("source").show(
                                &mut columns[0],
                                |ui| {
                                    ui.add(
                                        TextEdit::multiline(&mut code).font(TextStyle::Monospace),
                                    );
                                },
                            );
                            ScrollArea::vertical().id_source("rendered").show(
                                &mut columns[1],
                                |ui| {
                                    egui_demo_lib::easy_mark::easy_mark(ui, &code);
                                },
                            );
                        });
                    });
                });
                // Render UI
                renderer.render(&mut gui);
            }
            Event::MainEventsCleared => {
                renderer.surface().window().request_redraw();
            }
            _ => (),
        }
    });
}

const CODE: &str = r#"
# Some markup
```
let mut gui = Gui::new(renderer.surface(), renderer.queue());
```

Vulkan(o) is hard, that I know...
"#;

struct SimpleGuiRenderer {
    #[allow(dead_code)]
    instance: Arc<Instance>,
    device: Arc<Device>,
    surface: Arc<Surface<Window>>,
    queue: Arc<Queue>,
    render_pass: Arc<RenderPass>,
    pipeline: Arc<GraphicsPipeline>,
    swap_chain: Arc<Swapchain<Window>>,
    final_images: Vec<Arc<ImageView<SwapchainImage<Window>>>>,
    recreate_swapchain: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
    vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
}

impl SimpleGuiRenderer {
    pub fn new(
        event_loop: &EventLoop<()>,
        window_size: [u32; 2],
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
        let render_pass = Self::create_render_pass(device.clone(), images[0].format().unwrap());
        let pipeline = Self::create_pipeline(device.clone(), render_pass.clone());

        let vertex_buffer = {
            CpuAccessibleBuffer::from_iter(
                queue.device().clone(),
                BufferUsage::all(),
                false,
                [
                    Vertex { position: [-0.5, -0.25], color: [1.0, 0.0, 0.0, 1.0] },
                    Vertex { position: [0.0, 0.5], color: [0.0, 1.0, 0.0, 1.0] },
                    Vertex { position: [0.25, -0.1], color: [0.0, 0.0, 1.0, 1.0] },
                ]
                .iter()
                .cloned(),
            )
            .expect("failed to create buffer")
        };

        Self {
            instance,
            device,
            surface,
            queue,
            render_pass,
            pipeline,
            swap_chain,
            final_images: images,
            previous_frame_end,
            recreate_swapchain: false,
            vertex_buffer,
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

    fn create_render_pass(device: Arc<Device>, format: Format) -> Arc<RenderPass> {
        vulkano::ordered_passes_renderpass!(
            device,
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: format,
                    samples: 1,
                }
            },
            passes: [
                { color: [color], depth_stencil: {}, input: [] }, // Draw what you want on this pass
                { color: [color], depth_stencil: {}, input: [] } // Gui render pass
            ]
        )
        .unwrap()
    }

    fn gui_pass(&self) -> Subpass {
        Subpass::from(self.render_pass.clone(), 1).unwrap()
    }

    fn create_pipeline(device: Arc<Device>, render_pass: Arc<RenderPass>) -> Arc<GraphicsPipeline> {
        let vs = vs::load(device.clone()).expect("failed to create shader module");
        let fs = fs::load(device.clone()).expect("failed to create shader module");

        GraphicsPipeline::start()
            .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            .input_assembly_state(InputAssemblyState::new())
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .render_pass(Subpass::from(render_pass, 0).unwrap())
            .build(device)
            .unwrap()
    }

    pub fn queue(&self) -> Arc<Queue> {
        self.queue.clone()
    }

    pub fn surface(&self) -> Arc<Surface<Window>> {
        self.surface.clone()
    }

    pub fn resize(&mut self) {
        self.recreate_swapchain = true;
    }

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

        let mut builder = AutoCommandBufferBuilder::primary(
            self.device.clone(),
            self.queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let dimensions = self.final_images[0].image().dimensions().width_height();
        let framebuffer = Framebuffer::new(self.render_pass.clone(), FramebufferCreateInfo {
            attachments: vec![self.final_images[image_num].clone()],
            ..Default::default()
        })
        .unwrap();

        // Begin render pipeline commands
        let clear_values = vec![[0.0, 1.0, 0.0, 1.0].into()];
        builder
            .begin_render_pass(
                framebuffer,
                SubpassContents::SecondaryCommandBuffers,
                clear_values,
            )
            .unwrap();

        // Render first draw pass
        let mut secondary_builder = AutoCommandBufferBuilder::secondary_graphics(
            self.device.clone(),
            self.queue.family(),
            CommandBufferUsage::MultipleSubmit,
            self.pipeline.subpass().clone(),
        )
        .unwrap();
        secondary_builder
            .bind_pipeline_graphics(self.pipeline.clone())
            .set_viewport(0, vec![Viewport {
                origin: [0.0, 0.0],
                dimensions: [dimensions[0] as f32, dimensions[1] as f32],
                depth_range: 0.0..1.0,
            }])
            .bind_vertex_buffers(0, self.vertex_buffer.clone())
            .draw(self.vertex_buffer.len() as u32, 1, 0, 0)
            .unwrap();
        let cb = secondary_builder.build().unwrap();
        builder.execute_commands(cb).unwrap();

        // Move on to next subpass for gui
        builder.next_subpass(SubpassContents::SecondaryCommandBuffers).unwrap();
        // Draw gui on subpass
        let cb = gui.draw_on_subpass_image(dimensions);
        builder.execute_commands(cb).unwrap();

        // Last end render pass
        builder.end_render_pass().unwrap();
        let command_buffer = builder.build().unwrap();
        let before_future = self.previous_frame_end.take().unwrap().join(acquire_future);
        let after_future = before_future.then_execute(self.queue.clone(), command_buffer).unwrap();

        // Finish render
        self.finish(after_future.boxed(), image_num);
    }

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

#[repr(C)]
#[derive(Default, Debug, Copy, Clone, Zeroable, Pod)]
struct Vertex {
    position: [f32; 2],
    color: [f32; 4],
}
vulkano::impl_vertex!(Vertex, position, color);

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
#version 450
layout(location = 0) in vec2 position;
layout(location = 1) in vec4 color;

layout(location = 0) out vec4 v_color;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    v_color = color;
}"
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
#version 450
layout(location = 0) in vec4 v_color;

layout(location = 0) out vec4 f_color;

void main() {
    f_color = v_color;
}"
    }
}
