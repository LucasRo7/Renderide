//! Shared initialization for queue file, mmap, and semaphore (Unix) or file-only (Windows).

use std::fs::{self, File, OpenOptions};
use std::path::PathBuf;

use memmap2::MmapMut;

use crate::queue::QueueOptions;
use crate::sem::{self, SemHandle};

/// Opens the queue file, mmaps it, and creates the semaphore (Unix) or just file+mmap (Windows).
/// Returns (file, mmap, sem_handle, file_path) for use by Subscriber or Publisher.
pub(super) fn open_queue_backing(
    options: &QueueOptions,
) -> (File, MmapMut, SemHandle, PathBuf) {
    let path = options.file_path();
    fs::create_dir_all(path.parent().unwrap()).ok();

    let storage_size = options.actual_storage_size() as u64;
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(&path)
        .expect("Failed to open queue file");

    file.set_len(storage_size)
        .expect("Failed to set file length");

    let mmap = unsafe { MmapMut::map_mut(&file).expect("Failed to mmap queue file") };

    let sem_name = options.memory_view_name.clone();
    let sem_handle = sem::open(&sem_name);

    (file, mmap, sem_handle, path)
}
