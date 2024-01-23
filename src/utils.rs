// Copyright (c) 2021 Okko Hakola
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;

use image::RgbaImage;
use vulkano::{
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        CopyBufferToImageInfo, PrimaryCommandBufferAbstract,
    },
    device::Queue,
    image::{view::ImageView, ImageCreateInfo, ImageType, ImageUsage},
    NonZeroDeviceSize, Validated, ValidationError, VulkanError,
};

use crate::allocator::Allocators;

#[derive(Debug)]
pub enum ImageCreationError<AllocErr: std::fmt::Debug> {
    Vulkan(Validated<VulkanError>),
    AllocateImage(AllocErr),
    AllocateBuffer(AllocErr),
    Validation(Box<ValidationError>),
    BadSize,
}

pub fn immutable_texture_from_bytes<Alloc>(
    queue: Arc<Queue>,
    allocators: Alloc,
    byte_data: &[u8],
    dimensions: [u32; 2],
    format: vulkano::format::Format,
) -> Result<Arc<ImageView>, ImageCreationError<Alloc::Error>>
where
    Alloc: Allocators,
{
    let mut cbb: AutoCommandBufferBuilder<
        vulkano::command_buffer::PrimaryAutoCommandBuffer<_>,
        StandardCommandBufferAllocator,
    > = AutoCommandBufferBuilder::primary(
        todo!(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .map_err(ImageCreationError::Vulkan)?;

    let Some(size) = NonZeroDeviceSize::new(byte_data.len().try_into().unwrap()) else {
        return Err(ImageCreationError::BadSize);
    };

    let texture_data_buffer =
        allocators.make_image_stage_buffer(size).map_err(ImageCreationError::AllocateBuffer)?;
    {
        let mut write = texture_data_buffer.write().unwrap();
        write.iter_mut().zip(byte_data.iter()).for_each(|(into, &from)| *into = from);
    }

    let texture = allocators
        .make_image(ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format,
            extent: [dimensions[0], dimensions[1], 1],
            usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
            ..Default::default()
        })
        .map_err(ImageCreationError::AllocateImage)?;

    cbb.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
        texture_data_buffer,
        texture.clone(),
    ))
    .map_err(ImageCreationError::Validation)?;

    let _fut = cbb.build().unwrap().execute(queue).unwrap();

    Ok(ImageView::new_default(texture).unwrap())
}

pub fn immutable_texture_from_file<Alloc>(
    queue: Arc<Queue>,
    allocators: Alloc,
    file_bytes: &[u8],
    format: vulkano::format::Format,
) -> Result<Arc<ImageView>, ImageCreationError<Alloc::Error>>
where
    Alloc: Allocators,
{
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
    immutable_texture_from_bytes(queue, allocators, &rgba, [dimensions.0, dimensions.1], format)
}
