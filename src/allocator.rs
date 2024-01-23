use std::{fmt::Debug, sync::Arc};

use vulkano::{
    buffer::{
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
        BufferUsage, Subbuffer,
    },
    device::Device,
    image::{Image, ImageCreateInfo},
    memory::{
        allocator::{
            AllocationCreateInfo, BumpAllocator, DeviceLayout, GenericMemoryAllocator,
            GenericMemoryAllocatorCreateInfo, MemoryTypeFilter, StandardMemoryAllocator,
        },
        DeviceAlignment,
    },
    NonZeroDeviceSize,
};

/// A collection of allocators needed for the integration.
pub trait Allocators {
    type Error: Debug;
    /// Make a short-lived subbuffer for uploading and drawing vertices and indicies.
    ///
    /// * Must be `HOST_VISIBLE` (sequential write), and compatible with `VERTEX_BUFER ` and `INDEX_BUFFER` usages.
    fn make_vertex_index_buffer(
        &mut self,
        device_layout: DeviceLayout,
    ) -> Result<Subbuffer<[u8]>, Self::Error>;
    /// Make a short-lived subbuffer for uploading images.
    ///
    /// * Must be `HOST_VISIBLE` (sequential write), and compatible with `TRANSFER_SRC` usage.
    fn make_image_stage_buffer(
        &mut self,
        bytes_len: NonZeroDeviceSize,
    ) -> Result<Subbuffer<[u8]>, Self::Error>;
    /// Make a long-lived image. Corresponds one-to-one with an egui image.
    ///
    /// * Must be usable with `R8G8_UNORM` and `R8G8B8A8_SRGB` color images.
    fn make_image(&mut self, info: ImageCreateInfo) -> Result<Arc<Image>, Self::Error>;
}
impl<Base: Allocators> Allocators for &mut Base {
    type Error = Base::Error;
    fn make_vertex_index_buffer(
        &mut self,
        device_layout: DeviceLayout,
    ) -> Result<Subbuffer<[u8]>, Self::Error> {
        (*self).make_vertex_index_buffer(device_layout)
    }
    fn make_image_stage_buffer(
        &mut self,
        bytes_len: NonZeroDeviceSize,
    ) -> Result<Subbuffer<[u8]>, Self::Error> {
        (*self).make_image_stage_buffer(bytes_len)
    }
    fn make_image(&mut self, info: ImageCreateInfo) -> Result<Arc<Image>, Self::Error> {
        (*self).make_image(info)
    }
}

/// May be shared with several instances through use of [`DefaultAllocators::share`].
pub struct DefaultAllocators {
    /// The internal allocator of image_stage + vertex_index, for `share`ing.
    /// Bump is used as we expect these allocations to last only a fraction of a second, and to come in bursts.
    pub bump: Arc<GenericMemoryAllocator<BumpAllocator>>,
    pub image_stage: SubbufferAllocator<GenericMemoryAllocator<BumpAllocator>>,
    pub vertex_index: SubbufferAllocator<GenericMemoryAllocator<BumpAllocator>>,
    pub images: Arc<StandardMemoryAllocator>,
}

impl DefaultAllocators {
    /// 5MiB is a practical upper limit to vertex/index size - Openning all windows in `demo_app` doesn't even reach it.
    /// That many windows open is hardly usable let alone a common usecase!
    const SMALL_VERTEX_BLOCK: u64 = 8 * 1024 * 1024;
    /// 8MiB is enough stage for a ~1500 square fullcolor image, or 2048 square font image to be uploaded,
    /// Plus a generous wiggle room.
    const SMALL_IMAGE_STAGE_BLOCK: u64 = 16 * 1024 * 1024;
    /// This is a hard thing to "default" - on it's own egui uses only a tiny amount of image memory,
    /// however many usecases involve loading user images which greatly increases that.
    const IMAGE_BLOCK: u64 = 32 * 1024 * 1024;

    /// Allocators providing a good default for most apps, optimizing for many meshes and a few images.
    /// Aside from this, it will work for any usecase at some loss of efficiency.
    ///
    /// If you require many images or are updating images frequently,
    /// consider populating the fields with a custom allocator with large image memory pools.
    pub fn new_default(device: Arc<Device>) -> Self {
        let bump = Arc::new(GenericMemoryAllocator::new(
            device.clone(),
            GenericMemoryAllocatorCreateInfo {
                // Use the same size for all.
                // Many of these types wont be touched, and thus won't actually be allocated.
                block_sizes: &device
                    .physical_device()
                    .memory_properties()
                    .memory_types
                    .iter()
                    .map(|_| Self::SMALL_IMAGE_STAGE_BLOCK + Self::SMALL_VERTEX_BLOCK)
                    .collect::<Vec<_>>(),
                // These are transient resources, and should *always* return their mem in a
                // sub-divide-able manner to the pool after they're destroyed.
                dedicated_allocation: false,
                ..Default::default()
            },
        ));
        // Smarter allocator for long-lived images
        let images = Arc::new(StandardMemoryAllocator::new(
            device.clone(),
            GenericMemoryAllocatorCreateInfo {
                // Use the same size for all.
                // Many of these types wont be touched, and thus won't actually be allocated.
                block_sizes: &device
                    .physical_device()
                    .memory_properties()
                    .memory_types
                    .iter()
                    .map(|_| Self::IMAGE_BLOCK)
                    .collect::<Vec<_>>(),
                dedicated_allocation: true,
                ..Default::default()
            },
        ));
        Self::default_from_allocs(bump, images)
    }
    fn default_from_allocs(
        bump: Arc<GenericMemoryAllocator<BumpAllocator>>,
        images: Arc<StandardMemoryAllocator>,
    ) -> Self {
        // Neither of these are hard limits and will grow transparently if needed but are good reasonable guesses
        // Low-overhead bump memory for short-lived staging buffers
        Self {
            image_stage: SubbufferAllocator::new(bump.clone(), SubbufferAllocatorCreateInfo {
                arena_size: Self::SMALL_IMAGE_STAGE_BLOCK,
                buffer_usage: BufferUsage::TRANSFER_SRC,
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            }),
            vertex_index: SubbufferAllocator::new(bump.clone(), SubbufferAllocatorCreateInfo {
                arena_size: Self::SMALL_VERTEX_BLOCK,
                buffer_usage: BufferUsage::VERTEX_BUFFER | BufferUsage::INDEX_BUFFER,
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            }),
            bump,
            images,
        }
    }
    pub fn share(&self) -> Self {
        Self::default_from_allocs(self.bump.clone(), self.images.clone())
    }
}

// Every member is sync, so we don't need exclusive access to allocate.
impl Allocators for &DefaultAllocators {
    type Error = ();

    fn make_vertex_index_buffer(
        &mut self,
        device_layout: DeviceLayout,
    ) -> Result<Subbuffer<[u8]>, Self::Error> {
        self.vertex_index.allocate(device_layout).map_err(|_| ())
    }

    fn make_image_stage_buffer(
        &mut self,
        bytes_len: NonZeroDeviceSize,
    ) -> Result<Subbuffer<[u8]>, Self::Error> {
        // Infallible, align of one can never overflow `DeviceSize`
        let layout = DeviceLayout::new(bytes_len, DeviceAlignment::MIN).unwrap();
        self.image_stage.allocate(layout).map_err(|_| ())
    }

    fn make_image(&mut self, info: ImageCreateInfo) -> Result<Arc<Image>, Self::Error> {
        Image::new(self.images.clone(), info, AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        })
        .map_err(|_| ())
    }
}

// Delegate to &DefaultAllocators impl.
impl Allocators for DefaultAllocators {
    // Forward Err to &DefaultAllocators
    type Error = <&'static DefaultAllocators as Allocators>::Error;

    fn make_vertex_index_buffer(
        &mut self,
        device_layout: DeviceLayout,
    ) -> Result<Subbuffer<[u8]>, Self::Error> {
        (&*self).make_vertex_index_buffer(device_layout)
    }

    fn make_image_stage_buffer(
        &mut self,
        bytes_len: NonZeroDeviceSize,
    ) -> Result<Subbuffer<[u8]>, Self::Error> {
        (&*self).make_image_stage_buffer(bytes_len)
    }

    fn make_image(&mut self, info: ImageCreateInfo) -> Result<Arc<Image>, Self::Error> {
        (&*self).make_image(info)
    }
}
