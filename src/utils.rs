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
    format::{B8G8R8A8Unorm, Format::R8G8B8A8Srgb},
    image::{Dimensions, ImageCreationError, ImageViewAccess, ImmutableImage, MipmapsCount},
};

pub fn texture_from_bgra_bytes(
    queue: Arc<Queue>,
    bytes: &[u8],
    dimensions: (u64, u64),
) -> Result<Arc<dyn ImageViewAccess + Send + Sync>, ImageCreationError> {
    let vko_dims = Dimensions::Dim2d { width: dimensions.0 as u32, height: dimensions.1 as u32 };
    let (texture, _tex_fut) = ImmutableImage::from_iter(
        bytes.iter().cloned(),
        vko_dims,
        MipmapsCount::One,
        B8G8R8A8Unorm,
        queue.clone(),
    )?;
    Ok(texture)
}

pub fn texture_from_file_bytes(
    queue: Arc<Queue>,
    bytes: &[u8],
) -> Result<Arc<dyn ImageViewAccess + Send + Sync>, ImageCreationError> {
    use image::GenericImageView;

    let img = image::load_from_memory(bytes).expect("Failed to load image from bytes");
    let rgba = img.as_rgba8().unwrap().to_owned();
    let dimensions = img.dimensions();
    let vko_dims = Dimensions::Dim2d { width: dimensions.0, height: dimensions.1 };
    let (texture, _tex_fut) = ImmutableImage::from_iter(
        rgba.into_raw().into_iter(),
        vko_dims,
        MipmapsCount::One,
        R8G8B8A8Srgb,
        queue.clone(),
    )?;
    Ok(texture)
}
