//! Cloudtoid.Interprocess-compatible queue for IPC with Resonite host.
//! Unix: file-backed mmap at /dev/shm. Windows: named CreateFileMapping (CT_IP_{name}), like zinterprocess.

mod backend;
mod sem;
mod circular_buffer;
mod publisher;
mod queue;
mod subscriber;

pub use publisher::Publisher;
pub use queue::{QueueFactory, QueueOptions};
pub use subscriber::Subscriber;
