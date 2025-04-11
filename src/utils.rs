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
    buffer::{AllocateBufferError, BufferCreateInfo, BufferUsage},
    device::Queue,
    image::{
        view::{ImageView, ImageViewCreateInfo},
        AllocateImageError, Image, ImageCreateInfo, ImageLayout, ImageType, ImageUsage,
    },
    memory::allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter},
    Validated, VulkanError,
};
use vulkano_taskgraph::{
    command_buffer::CopyBufferToImageInfo,
    descriptor_set::SampledImageId,
    graph::ExecuteError,
    resource::{AccessTypes, Flight, HostAccessType, ImageLayoutType, Resources},
    Id,
};

#[derive(Debug)]
pub enum ImageCreationError {
    Vulkan(Validated<VulkanError>),
    AllocateBuffer(Validated<AllocateBufferError>),
    AllocateImage(Validated<AllocateImageError>),
    ExecuteError(ExecuteError),
}

pub fn immutable_texture_from_bytes<W: 'static + ?Sized>(
    queue: Arc<Queue>,
    resources: Arc<Resources>,
    flight_id: Id<Flight>,
    byte_data: &[u8],
    dimensions: [u32; 2],
    format: vulkano::format::Format,
) -> Result<(Id<Image>, SampledImageId), ImageCreationError> {
    let texture_data_buffer = resources
        .create_buffer(
            BufferCreateInfo { usage: BufferUsage::TRANSFER_SRC, ..Default::default() },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            DeviceLayout::new_unsized::<[u8]>(byte_data.len() as u64).unwrap(),
        )
        .map_err(ImageCreationError::AllocateBuffer)?;

    let texture_id = resources
        .create_image(
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

    let bcx = resources.bindless_context().unwrap();
    let image = resources.image(texture_id).unwrap().image().clone();
    let image_view = ImageView::new(image.clone(), ImageViewCreateInfo::from_image(&image))
        .map_err(ImageCreationError::Vulkan)?;

    let sampled_image_id = bcx.global_set().add_sampled_image(image_view, ImageLayout::General);

    let flight = resources.flight(flight_id).unwrap();
    flight.wait(None).unwrap();

    unsafe {
        vulkano_taskgraph::execute(
            &queue.clone(),
            &resources.clone(),
            flight_id,
            |builder, task_context| {
                let write_buffer = task_context.write_buffer::<[u8]>(texture_data_buffer, ..)?;
                write_buffer.copy_from_slice(byte_data);

                builder
                    .copy_buffer_to_image(&CopyBufferToImageInfo {
                        src_buffer: texture_data_buffer,
                        dst_image: texture_id,
                        ..Default::default()
                    })
                    .unwrap();

                Ok(())
            },
            [(texture_data_buffer, HostAccessType::Write)],
            [(texture_data_buffer, AccessTypes::COPY_TRANSFER_READ)],
            [(texture_id, AccessTypes::COPY_TRANSFER_WRITE, ImageLayoutType::Optimal)],
        )
    }
    .map_err(ImageCreationError::ExecuteError)?;

    let flight = resources.flight(flight_id).unwrap();
    flight.wait(None).unwrap();

    Ok((texture_id, sampled_image_id))
}

#[cfg(feature = "image")]
pub fn immutable_texture_from_file<W: 'static + ?Sized>(
    queue: Arc<Queue>,
    resources: Arc<Resources>,
    flight_id: Id<Flight>,
    file_bytes: &[u8],
    format: vulkano::format::Format,
) -> Result<(Id<Image>, SampledImageId), ImageCreationError> {
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
    immutable_texture_from_bytes::<W>(
        queue,
        resources,
        flight_id,
        &rgba,
        [dimensions.0, dimensions.1],
        format,
    )
}
