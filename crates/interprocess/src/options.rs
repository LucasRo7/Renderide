//! Configuration for opening a shared-memory queue.

use std::path::{Path, PathBuf};

/// Default directory for queue backing files on Unix (typically tmpfs at `/dev/shm`).
pub const DEFAULT_MEMORY_DIR: &str = "/dev/shm/.cloudtoid/interprocess/mmf";

/// Legacy alias for [`DEFAULT_MEMORY_DIR`].
pub const MEMORY_FILE_PATH: &str = DEFAULT_MEMORY_DIR;

/// Options for creating a [`crate::Publisher`] or [`crate::Subscriber`].
#[derive(Clone)]
pub struct QueueOptions {
    /// Logical queue name (maps to `{dir}/{name}.qu` on Unix and `CT_IP_{name}` on Windows).
    pub memory_view_name: String,
    /// Directory containing `.qu` files on Unix; ignored for the default Windows named-mapping backend.
    pub path: PathBuf,
    /// Ring buffer capacity in bytes (user data only; excludes [`crate::layout::QueueHeader`]).
    pub capacity: i64,
    /// When `true`, remove the backing file (Unix) when the handle is dropped.
    pub destroy_on_dispose: bool,
}

impl QueueOptions {
    const MIN_CAPACITY: i64 = 17;

    /// Validates `capacity` and returns an error message if invalid.
    fn validate_capacity(capacity: i64) -> Result<(), String> {
        if capacity <= Self::MIN_CAPACITY {
            return Err(format!(
                "capacity must be greater than {} (got {capacity})",
                Self::MIN_CAPACITY
            ));
        }
        if capacity % 8 != 0 {
            return Err(format!(
                "capacity must be a multiple of 8 bytes (got {capacity})"
            ));
        }
        Ok(())
    }

    /// Builds options with [`DEFAULT_MEMORY_DIR`] and `destroy_on_dispose = false`.
    pub fn new(queue_name: &str, capacity: i64) -> Result<Self, String> {
        Self::validate_capacity(capacity)?;
        Ok(Self {
            memory_view_name: queue_name.to_string(),
            path: PathBuf::from(MEMORY_FILE_PATH),
            capacity,
            destroy_on_dispose: false,
        })
    }

    /// Same as [`Self::new`] but controls whether the backing file is removed on drop (Unix).
    pub fn with_destroy(
        queue_name: &str,
        capacity: i64,
        destroy_on_dispose: bool,
    ) -> Result<Self, String> {
        Self::validate_capacity(capacity)?;
        Ok(Self {
            memory_view_name: queue_name.to_string(),
            path: PathBuf::from(MEMORY_FILE_PATH),
            capacity,
            destroy_on_dispose,
        })
    }

    /// Full control over the backing directory.
    pub fn with_path(
        queue_name: &str,
        path: impl AsRef<Path>,
        capacity: i64,
    ) -> Result<Self, String> {
        Self::validate_capacity(capacity)?;
        Ok(Self {
            memory_view_name: queue_name.to_string(),
            path: path.as_ref().to_path_buf(),
            capacity,
            destroy_on_dispose: false,
        })
    }

    /// Full control over directory and `destroy_on_dispose`.
    pub fn with_path_and_destroy(
        queue_name: &str,
        path: impl AsRef<Path>,
        capacity: i64,
        destroy_on_dispose: bool,
    ) -> Result<Self, String> {
        Self::validate_capacity(capacity)?;
        Ok(Self {
            memory_view_name: queue_name.to_string(),
            path: path.as_ref().to_path_buf(),
            capacity,
            destroy_on_dispose,
        })
    }

    /// Total file / mapping size: header + ring capacity.
    pub fn actual_storage_size(&self) -> i64 {
        crate::layout::BUFFER_BYTE_OFFSET as i64 + self.capacity
    }

    /// Path to the `.qu` backing file on Unix.
    pub fn file_path(&self) -> PathBuf {
        self.path.join(format!("{}.qu", self.memory_view_name))
    }

    /// POSIX semaphore name (`/ct.ip.{memory_view_name}`).
    pub fn posix_semaphore_name(&self) -> String {
        format!("/ct.ip.{}", self.memory_view_name)
    }
}
