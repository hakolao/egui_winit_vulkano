// Copyright (c) 2021 Okko Hakola
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;

use vulkano::{
    device::Queue,
    image::{
        view::ImageView, ImageCreationError, ImageDimensions, ImageViewAbstract, ImmutableImage,
        MipmapsCount,
    },
};

pub fn texture_from_bytes(
    queue: Arc<Queue>,
    byte_data: &[u8],
    dimensions: (u64, u64),
    format: vulkano::format::Format,
) -> Result<Arc<dyn ImageViewAbstract + Send + Sync + 'static>, ImageCreationError> {
    let vko_dims = ImageDimensions::Dim2d {
        width: dimensions.0 as u32,
        height: dimensions.1 as u32,
        array_layers: 1,
    };

    let (texture, _tex_fut) = ImmutableImage::from_iter(
        byte_data.iter().cloned(),
        vko_dims,
        MipmapsCount::One,
        format,
        queue.clone(),
    )?;
    Ok(ImageView::new(texture).unwrap())
}

pub fn texture_from_file(
    queue: Arc<Queue>,
    file_bytes: &[u8],
    format: vulkano::format::Format,
) -> Result<Arc<dyn ImageViewAbstract + Send + Sync + 'static>, ImageCreationError> {
    use image::GenericImageView;

    let img = image::load_from_memory(file_bytes).expect("Failed to load image from bytes");
    let rgba = img.as_rgba8().unwrap().to_owned();
    let dimensions = img.dimensions();
    let vko_dims =
        ImageDimensions::Dim2d { width: dimensions.0, height: dimensions.1, array_layers: 1 };
    let (texture, _tex_fut) = ImmutableImage::from_iter(
        rgba.into_raw().into_iter(),
        vko_dims,
        MipmapsCount::One,
        format,
        queue.clone(),
    )?;
    Ok(ImageView::new(texture).unwrap())
}
