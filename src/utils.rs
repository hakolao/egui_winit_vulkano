use std::sync::Arc;
use vulkano::device::Queue;
use vulkano::format::B8G8R8A8Unorm;
use vulkano::format::Format::R8G8B8A8Srgb;
use vulkano::image::{
    Dimensions, ImageCreationError, ImageViewAccess, ImmutableImage, MipmapsCount,
};

pub fn texture_from_bgra_bytes(
    queue: Arc<Queue>,
    bytes: &[u8],
    dimensions: (u64, u64),
) -> Result<Arc<dyn ImageViewAccess + Send + Sync>, ImageCreationError> {
    let vko_dims = Dimensions::Dim2d {
        width: dimensions.0 as u32,
        height: dimensions.1 as u32,
    };
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
    let img = image::load_from_memory(bytes).expect("Failed to load image from bytes");
    let rgba = img.as_rgba8().unwrap().to_owned();
    let dimensions = img.dimensions();
    let vko_dims = Dimensions::Dim2d {
        width: dimensions.0,
        height: dimensions.1,
    };
    let (texture, _tex_fut) = ImmutableImage::from_iter(
        rgba.into_raw().into_iter(),
        vko_dims,
        MipmapsCount::One,
        R8G8B8A8Srgb,
        queue.clone(),
    )?;
    Ok(texture)
}
