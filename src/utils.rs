// Copyright (c) 2021 Okko Hakola
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;

#[cfg(feature = "image")]
use image::RgbaImage;
use vulkano::{
    buffer::{AllocateBufferError, Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo,
        PrimaryCommandBufferAbstract,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{Device, Queue},
    image::{view::ImageView, AllocateImageError, Image, ImageCreateInfo, ImageType, ImageUsage},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    Validated, ValidationError, VulkanError,
};

#[derive(Debug)]
pub enum ImageCreationError {
    Vulkan(Validated<VulkanError>),
    AllocateImage(Validated<AllocateImageError>),
    AllocateBuffer(Validated<AllocateBufferError>),
    Validation(Box<ValidationError>),
}

pub fn immutable_texture_from_bytes(
    allocators: &Allocators,
    queue: Arc<Queue>,
    byte_data: &[u8],
    dimensions: [u32; 2],
    format: vulkano::format::Format,
) -> Result<Arc<ImageView>, ImageCreationError> {
    let mut cbb = AutoCommandBufferBuilder::primary(
        &allocators.command_buffer,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .map_err(ImageCreationError::Vulkan)?;

    let texture_data_buffer = Buffer::from_iter(
        allocators.memory.clone(),
        BufferCreateInfo { usage: BufferUsage::TRANSFER_SRC, ..Default::default() },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        byte_data.iter().cloned(),
    )
    .map_err(ImageCreationError::AllocateBuffer)?;

    let texture = Image::new(
        allocators.memory.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format,
            extent: [dimensions[0], dimensions[1], 1],
            usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
    )
    .map_err(ImageCreationError::AllocateImage)?;

    cbb.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
        texture_data_buffer,
        texture.clone(),
    ))
    .map_err(ImageCreationError::Validation)?;

    let _fut = cbb.build().unwrap().execute(queue).unwrap();

    Ok(ImageView::new_default(texture).unwrap())
}

#[cfg(feature = "image")]
pub fn immutable_texture_from_file(
    allocators: &Allocators,
    queue: Arc<Queue>,
    file_bytes: &[u8],
    format: vulkano::format::Format,
) -> Result<Arc<ImageView>, ImageCreationError> {
    use image::GenericImageView;

    let img = image::load_from_memory(file_bytes).expect("Failed to load image from bytes");
    let rgba = if let Some(rgba) = img.as_rgba8() {
        rgba.to_owned().to_vec()
    } else {
        // Convert rgb to rgba
        let rgb = img.as_rgb8().unwrap().to_owned();
        let mut raw_data = vec![];
        for val in rgb.chunks(3) {
            raw_data.push(val[0]);
            raw_data.push(val[1]);
            raw_data.push(val[2]);
            raw_data.push(255);
        }
        let new_rgba = RgbaImage::from_raw(rgb.width(), rgb.height(), raw_data).unwrap();
        new_rgba.to_vec()
    };
    let dimensions = img.dimensions();
    immutable_texture_from_bytes(allocators, queue, &rgba, [dimensions.0, dimensions.1], format)
}

pub struct Allocators {
    pub memory: Arc<StandardMemoryAllocator>,
    pub descriptor_set: StandardDescriptorSetAllocator,
    pub command_buffer: StandardCommandBufferAllocator,
}

impl Allocators {
    pub fn new_default(device: &Arc<Device>) -> Self {
        Self {
            memory: Arc::new(StandardMemoryAllocator::new_default(device.clone())),
            descriptor_set: StandardDescriptorSetAllocator::new(device.clone(), Default::default()),
            command_buffer: StandardCommandBufferAllocator::new(
                device.clone(),
                StandardCommandBufferAllocatorCreateInfo {
                    secondary_buffer_count: 32,
                    ..Default::default()
                },
            ),
        }
    }
}
