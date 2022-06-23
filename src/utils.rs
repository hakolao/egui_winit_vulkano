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
    device::Queue,
    image::{
        immutable::ImmutableImageCreationError, view::ImageView, ImageDimensions,
        ImageViewAbstract, ImmutableImage, MipmapsCount,
    },
};

pub fn immutable_texture_from_bytes(
    queue: Arc<Queue>,
    byte_data: &[u8],
    dimensions: [u32; 2],
    format: vulkano::format::Format,
) -> Result<Arc<dyn ImageViewAbstract + Send + Sync + 'static>, ImmutableImageCreationError> {
    let vko_dims =
        ImageDimensions::Dim2d { width: dimensions[0], height: dimensions[1], array_layers: 1 };

    let (texture, _tex_fut) = ImmutableImage::from_iter(
        byte_data.iter().cloned(),
        vko_dims,
        MipmapsCount::One,
        format,
        queue,
    )?;

    Ok(ImageView::new_default(texture).unwrap())
}

pub fn immutable_texture_from_file(
    queue: Arc<Queue>,
    file_bytes: &[u8],
    format: vulkano::format::Format,
) -> Result<Arc<dyn ImageViewAbstract + Send + Sync + 'static>, ImmutableImageCreationError> {
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
    immutable_texture_from_bytes(queue, &rgba, [dimensions.0, dimensions.1], format)
}
