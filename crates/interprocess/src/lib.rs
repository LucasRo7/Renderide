//! Cloudtoid.Interprocess-compatible queue for IPC with Resonite host.
//! Uses shared memory + named semaphores (POSIX on Unix, Windows on Windows).
//! Subscriber uses polling/sleep when empty (zinterprocess style).

mod backend;
mod sem;
mod circular_buffer;
mod publisher;
mod queue;
mod subscriber;

pub use publisher::Publisher;
pub use queue::{QueueFactory, QueueOptions};
pub use subscriber::Subscriber;
