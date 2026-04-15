//! Host–renderer IPC: Cloudtoid ring-buffer queues and memory-mapped large payloads.
//!
//! Layout: [`connection`] (CLI `-QueueName` / `-QueueCapacity`, queue naming); [`dual_queue`]
//! ([`DualQueueIpc`]); [`shared_memory`] ([`SharedMemoryAccessor`], plus `bounds` / `naming` helpers
//! and platform `SharedMemoryView` modules).

pub mod connection;
mod dual_queue;
mod shared_memory;

pub use dual_queue::DualQueueIpc;
pub use shared_memory::{
    compose_memory_view_name, SharedMemoryAccessor, RENDERIDE_INTERPROCESS_DIR_ENV,
};
