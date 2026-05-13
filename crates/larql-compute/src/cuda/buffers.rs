//! GPU buffer cache — device memory pool for reusable allocations.
//!
//! Phase 0 implementation: simple Vec-based pool with size-class reuse.

use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaContext, CudaSlice};

/// A single pooled buffer with its size class.
struct PooledBuffer {
    size_class: usize,
    buffer: CudaSlice<u8>,
}

/// Cached GPU buffer pool. Thread-safe via `Mutex`.
pub struct CudaBufferCache {
    device: Arc<CudaContext>,
    pool: Mutex<Vec<PooledBuffer>>,
}

impl CudaBufferCache {
    /// Create a new empty buffer cache bound to the given device.
    pub fn new(device: &Arc<CudaContext>) -> Self {
        Self {
            device: device.clone(),
            pool: Mutex::new(Vec::new()),
        }
    }

    /// Get a buffer of at least `size` bytes, allocating if necessary.
    ///
    /// Tries the pool for a matching size-class buffer first.
    /// If none available, allocates a new one on the device.
    pub fn get_or_alloc(&self, size: usize) -> Option<CudaSlice<u8>> {
        let size_class = size.next_multiple_of(256);

        // Try pool reuse first
        if let Ok(mut pool) = self.pool.lock() {
            if let Some(idx) = pool.iter().position(|b| b.size_class == size_class) {
                let pooled = pool.swap_remove(idx);
                return Some(pooled.buffer);
            }
        }

        // Allocate a new buffer on the default stream
        let stream = self.device.default_stream();
        stream.alloc_zeros::<u8>(size_class).ok()
    }

    /// Return a buffer to the pool for reuse.
    pub fn return_buffer(&self, buf: CudaSlice<u8>) {
        let size_class = buf.len().next_multiple_of(256);
        if let Ok(mut pool) = self.pool.lock() {
            pool.push(PooledBuffer { size_class, buffer: buf });
        }
    }

    /// Number of buffers currently in the pool.
    pub fn len(&self) -> usize {
        self.pool.lock().map(|p| p.len()).unwrap_or(0)
    }

    /// Whether the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}