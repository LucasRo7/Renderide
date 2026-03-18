//! Shared memory backing: file-backed mmap on Unix, named CreateFileMapping on Windows.
//! Windows naming matches Cloudtoid/zinterprocess: CT_IP_{name}.

use std::path::PathBuf;

use crate::queue::QueueOptions;
use crate::sem::{self, SemHandle};

/// Platform-agnostic backing for the queue. Provides raw pointer access to the mapped region.
pub(super) struct MemoryBacking {
    #[cfg(unix)]
    inner: UnixBacking,
    #[cfg(windows)]
    inner: WindowsBacking,
}

#[cfg(unix)]
struct UnixBacking {
    _file: std::fs::File,
    mmap: memmap2::MmapMut,
    file_path: PathBuf,
}

#[cfg(windows)]
struct WindowsBacking {
    map_handle: windows_sys::Win32::Foundation::HANDLE,
    view: windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS,
    len: usize,
}

impl MemoryBacking {
    pub fn as_ptr(&self) -> *const u8 {
        #[cfg(unix)]
        {
            self.inner.mmap.as_ptr()
        }
        #[cfg(windows)]
        {
            self.inner.view.Value as *const u8
        }
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        #[cfg(unix)]
        {
            self.inner.mmap.as_mut_ptr()
        }
        #[cfg(windows)]
        {
            self.inner.view.Value as *mut u8
        }
    }

    pub fn len(&self) -> usize {
        #[cfg(unix)]
        {
            self.inner.mmap.len()
        }
        #[cfg(windows)]
        {
            self.inner.len
        }
    }

    /// Path for destroy_on_dispose (Unix only). None on Windows (named mapping cleans up on CloseHandle).
    pub fn file_path(&self) -> Option<PathBuf> {
        #[cfg(unix)]
        {
            Some(self.inner.file_path.clone())
        }
        #[cfg(windows)]
        {
            None
        }
    }

    /// Whether we should try to remove a file on dispose. Windows: no.
    pub fn has_file_to_remove(&self) -> bool {
        #[cfg(unix)]
        {
            true
        }
        #[cfg(windows)]
        {
            false
        }
    }
}

#[cfg(unix)]
impl Drop for UnixBacking {
    fn drop(&mut self) {}
}

#[cfg(windows)]
impl Drop for WindowsBacking {
    fn drop(&mut self) {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::UnmapViewOfFile;

        if !self.view.Value.is_null() {
            unsafe {
                UnmapViewOfFile(self.view);
            }
        }
        if self.map_handle != 0 && self.map_handle != -1 {
            unsafe {
                CloseHandle(self.map_handle);
            }
        }
    }
}

/// Opens the queue backing. Returns (MemoryBacking, SemHandle).
pub(super) fn open_queue_backing(options: &QueueOptions) -> (MemoryBacking, SemHandle) {
    #[cfg(unix)]
    {
        open_queue_backing_unix(options)
    }
    #[cfg(windows)]
    {
        open_queue_backing_windows(options)
    }
}

#[cfg(unix)]
fn open_queue_backing_unix(options: &QueueOptions) -> (MemoryBacking, SemHandle) {
    use std::fs::{self, File, OpenOptions};

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

    let mmap = unsafe { memmap2::MmapMut::map_mut(&file).expect("Failed to mmap queue file") };

    let sem_handle = sem::open(&options.memory_view_name);

    let backing = MemoryBacking {
        inner: UnixBacking {
            _file: file,
            mmap,
            file_path: path,
        },
    };

    (backing, sem_handle)
}

#[cfg(windows)]
fn open_queue_backing_windows(options: &QueueOptions) -> (MemoryBacking, SemHandle) {
    use std::ffi::OsStr;
    use std::os::windows::ffi::OsStrExt;

    use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
    use windows_sys::Win32::System::Memory::{MapViewOfFile, FILE_MAP_ALL_ACCESS};

    const MAP_NAME_PREFIX: &str = "CT_IP_";
    let name = format!("{}{}", MAP_NAME_PREFIX, options.memory_view_name);
    let storage_size = options.actual_storage_size() as usize;

    let name_wide: Vec<u16> = OsStr::new(&name)
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();

    let map_handle = create_or_open_file_mapping(&name_wide, storage_size);

    let view = unsafe {
        MapViewOfFile(
            map_handle,
            FILE_MAP_ALL_ACCESS,
            0,
            0,
            storage_size,
        )
    };

    if view.Value.is_null() {
        unsafe { windows_sys::Win32::Foundation::CloseHandle(map_handle) };
        panic!("MapViewOfFile failed for queue: {}", options.memory_view_name);
    }

    let sem_handle = sem::open(&options.memory_view_name);

    let backing = MemoryBacking {
        inner: WindowsBacking {
            map_handle,
            view,
            len: storage_size,
        },
    };

    (backing, sem_handle)
}

#[cfg(windows)]
fn create_or_open_file_mapping(name: &[u16], size: usize) -> windows_sys::Win32::Foundation::HANDLE {
    use std::ptr::null;
    use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
    use windows_sys::Win32::System::Memory::{
        CreateFileMappingW, OpenFileMappingW, FILE_MAP_ALL_ACCESS, PAGE_READWRITE,
    };

    let mut wait_retries: usize = 14;
    let mut wait_sleep_ms: u64 = 0;

    loop {
        let handle = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                null(),
                PAGE_READWRITE,
                (size >> 32) as u32,
                (size & 0xFFFF_FFFF) as u32,
                name.as_ptr(),
            )
        };

        if handle != 0 && handle != -1 {
            return handle;
        }

        let handle = unsafe {
            OpenFileMappingW(
                FILE_MAP_ALL_ACCESS,
                0,
                name.as_ptr(),
            )
        };

        if handle != 0 && handle != -1 {
            return handle;
        }

        wait_retries = match wait_retries.checked_sub(1) {
            Some(n) => n,
            None => panic!("Failed to create or open file mapping after retries"),
        };

        if wait_sleep_ms == 0 {
            wait_sleep_ms = 10;
        } else {
            std::thread::sleep(std::time::Duration::from_millis(wait_sleep_ms));
            wait_sleep_ms = (wait_sleep_ms * 2).min(1000);
        }
    }
}
