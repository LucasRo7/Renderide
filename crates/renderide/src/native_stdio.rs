//! Redirect native **stdout** and **stderr** into the Renderide file logger on Unix and Windows.
//!
//! Vulkan validation layers and **spirv-val** often emit via **`printf`** (stdout) and/or
//! **`fprintf(stderr, …)`**. WGPU’s instance flags do not control whether users enable layers via
//! `VK_INSTANCE_LAYERS`, so the renderer installs forwarding **unconditionally** after file logging
//! starts (see [`crate::app::run`]).
//!
//! OpenXR runtimes use the same native paths; [`crate::xr::bootstrap::init_wgpu_openxr`] also calls
//! [`ensure_stdio_forwarded_to_logger`] for entry points that skip `run` (idempotent via [`Once`]).
//!
//! - **Unix:** `pipe` + `dup2` per stream (`STDOUT_FILENO` / `STDERR_FILENO`).
//! - **Windows:** `CreatePipe` + `SetStdHandle(STD_OUTPUT_HANDLE / STD_ERROR_HANDLE, …)`.
//!
//! The readers use [`logger::try_log`] (non-blocking lock + append fallback) so they cannot deadlock
//! with the main thread on the global logger mutex, and read in chunks so a missing newline cannot
//! fill the pipe and block writers.
//!
//! On other targets this module is a no-op.
//!
//! Avoid enabling the logger’s **mirror-to-stderr** option together with this redirect: mirrored
//! lines would be written back into the pipe and re-logged.

use std::sync::Once;

use logger::LogLevel;

static INSTALL: Once = Once::new();

/// Ensures process **stdout** and **stderr** are forwarded to [`logger`] and no longer write to the
/// original terminal streams. Idempotent.
pub(crate) fn ensure_stdio_forwarded_to_logger() {
    INSTALL.call_once(|| {
        #[cfg(unix)]
        {
            if let Err(e) = try_redirect_unix_stream(libc::STDERR_FILENO, "renderide-stderr") {
                logger::warn!("Native stderr could not be redirected to log file: {e}");
            }
            if let Err(e) = try_redirect_unix_stream(libc::STDOUT_FILENO, "renderide-stdout") {
                logger::warn!("Native stdout could not be redirected to log file: {e}");
            }
        }
        #[cfg(windows)]
        {
            if let Err(e) = try_redirect_windows_stream(
                windows_sys::Win32::System::Console::STD_ERROR_HANDLE,
                "renderide-stderr",
            ) {
                logger::warn!("Native stderr could not be redirected to log file: {e}");
            }
            if let Err(e) = try_redirect_windows_stream(
                windows_sys::Win32::System::Console::STD_OUTPUT_HANDLE,
                "renderide-stdout",
            ) {
                logger::warn!("Native stdout could not be redirected to log file: {e}");
            }
        }
    });
}

#[cfg(unix)]
fn try_redirect_unix_stream(target_fd: i32, thread_name: &'static str) -> Result<(), String> {
    use std::thread;

    unsafe {
        let mut fds = [0i32; 2];
        if libc::pipe(fds.as_mut_ptr()) != 0 {
            return Err(format!("pipe: {}", std::io::Error::last_os_error()));
        }
        let rfd = fds[0];
        let wfd = fds[1];

        for fd in [rfd, wfd] {
            let flags = libc::fcntl(fd, libc::F_GETFD);
            if flags >= 0 {
                libc::fcntl(fd, libc::F_SETFD, flags | libc::FD_CLOEXEC);
            }
        }

        let saved = libc::dup(target_fd);
        if saved < 0 {
            libc::close(rfd);
            libc::close(wfd);
            return Err(format!(
                "dup({target_fd}): {}",
                std::io::Error::last_os_error()
            ));
        }

        if libc::dup2(wfd, target_fd) < 0 {
            let e = std::io::Error::last_os_error();
            libc::close(rfd);
            libc::close(wfd);
            libc::close(saved);
            return Err(format!("dup2(pipe -> fd {target_fd}): {e}"));
        }
        libc::close(wfd);

        let spawn = thread::Builder::new()
            .name(thread_name.into())
            .spawn(move || forward_pipe_lines_to_logger_unix(rfd));

        match spawn {
            Ok(_) => {
                libc::close(saved);
                Ok(())
            }
            Err(e) => {
                let _ = libc::dup2(saved, target_fd);
                libc::close(rfd);
                libc::close(saved);
                Err(format!("thread spawn: {e}"))
            }
        }
    }
}

#[cfg(windows)]
fn try_redirect_windows_stream(std_handle: u32, thread_name: &'static str) -> Result<(), String> {
    use std::fs::File;
    use std::os::windows::io::{FromRawHandle, OwnedHandle};
    use std::ptr;
    use std::thread;

    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Console::{GetStdHandle, SetStdHandle};
    use windows_sys::Win32::System::Pipes::CreatePipe;

    unsafe {
        let mut read_h: HANDLE = INVALID_HANDLE_VALUE;
        let mut write_h: HANDLE = INVALID_HANDLE_VALUE;
        if CreatePipe(&mut read_h, &mut write_h, ptr::null(), 0) == 0 {
            return Err(format!("CreatePipe: {}", std::io::Error::last_os_error()));
        }

        let old = GetStdHandle(std_handle);
        if old.is_null() || old == INVALID_HANDLE_VALUE {
            CloseHandle(read_h);
            CloseHandle(write_h);
            return Err(format!(
                "GetStdHandle({std_handle}): {}",
                std::io::Error::last_os_error()
            ));
        }

        if SetStdHandle(std_handle, write_h) == 0 {
            CloseHandle(read_h);
            CloseHandle(write_h);
            return Err(format!("SetStdHandle: {}", std::io::Error::last_os_error()));
        }

        let read_owned = OwnedHandle::from_raw_handle(read_h);

        let spawn = thread::Builder::new()
            .name(thread_name.into())
            .spawn(move || {
                let f = File::from(read_owned);
                forward_pipe_lines_to_logger_impl(f);
            });

        match spawn {
            Ok(_) => {
                let _ = CloseHandle(old);
                Ok(())
            }
            Err(e) => {
                let _ = SetStdHandle(std_handle, old);
                CloseHandle(write_h);
                Err(format!("thread spawn: {e}"))
            }
        }
    }
}

#[cfg(any(unix, windows))]
fn forward_pipe_lines_to_logger_impl<R: std::io::Read>(mut reader: R) {
    let mut pending = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        match reader.read(&mut chunk) {
            Ok(0) => {
                if !pending.is_empty() {
                    emit_stdio_line(&pending, LogLevel::Info);
                }
                break;
            }
            Ok(n) => {
                pending.extend_from_slice(&chunk[..n]);
                while let Some(pos) = pending.iter().position(|&b| b == b'\n') {
                    let line: Vec<u8> = pending.drain(..pos).collect();
                    if !pending.is_empty() && pending[0] == b'\n' {
                        pending.remove(0);
                    }
                    emit_stdio_line(&line, LogLevel::Info);
                }
            }
            Err(e) => {
                let _ = logger::try_log(
                    LogLevel::Debug,
                    format_args!("stdio forward read ended: {e}"),
                );
                break;
            }
        }
    }
}

#[cfg(unix)]
fn forward_pipe_lines_to_logger_unix(rfd: i32) {
    use std::fs::File;
    use std::os::unix::io::FromRawFd;

    let f = unsafe { File::from_raw_fd(rfd) };
    forward_pipe_lines_to_logger_impl(f);
}

#[cfg(any(unix, windows))]
fn emit_stdio_line(line: &[u8], level: LogLevel) {
    let t = String::from_utf8_lossy(line).trim().to_string();
    if t.is_empty() {
        return;
    }
    let _ = logger::try_log(level, format_args!("{t}"));
}
