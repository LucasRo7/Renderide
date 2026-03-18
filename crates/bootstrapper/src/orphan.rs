//! Orphan process cleanup from previous crashed runs.
//! Uses the PID file written by the bootstrapper.

use std::fs;
use std::io::Write;

use crate::paths;

/// Kills orphaned Host/renderer processes from a previous crashed run.
/// Call before spawning anything. Reads PIDs from the PID file; on Unix uses SIGTERM,
/// on Windows uses TerminateProcess.
pub fn kill_orphans() {
    let path = paths::pid_file_path();
    let Ok(contents) = fs::read_to_string(&path) else {
        return;
    };
    let _ = fs::remove_file(&path);

    for line in contents.lines() {
        let pid = match line
            .strip_prefix("host:")
            .or_else(|| line.strip_prefix("renderer:"))
        {
            Some(rest) => rest.trim().parse::<u32>().ok(),
            None => continue,
        };
        let Some(pid) = pid else {
            continue;
        };
        if process_exists(pid) {
            logger::info!("Killing orphan process {} from previous run", pid);
            kill_process(pid);
        }
    }
}

#[cfg(unix)]
fn process_exists(pid: u32) -> bool {
    unsafe { libc::kill(pid as i32, 0) == 0 }
}

#[cfg(unix)]
fn kill_process(pid: u32) {
    let _ = unsafe { libc::kill(pid as i32, libc::SIGTERM) };
}

#[cfg(windows)]
fn process_exists(pid: u32) -> bool {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION};

    let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
    if handle != 0 && handle != -1 {
        unsafe { CloseHandle(handle) };
        true
    } else {
        false
    }
}

#[cfg(windows)]
fn kill_process(pid: u32) {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{OpenProcess, TerminateProcess, PROCESS_TERMINATE};

    let handle = unsafe { OpenProcess(PROCESS_TERMINATE, 0, pid) };
    if handle != 0 && handle != -1 {
        unsafe {
            TerminateProcess(handle, 1);
            CloseHandle(handle);
        }
    }
}

/// Appends a PID entry to the PID file.
pub fn write_pid_file(pid: u32, kind: &str) {
    let path = paths::pid_file_path();
    match fs::OpenOptions::new().create(true).append(true).open(&path) {
        Ok(mut f) => {
            if let Err(e) = writeln!(f, "{}:{}", kind, pid) {
                logger::error!("Failed to write PID file: {}", e);
            }
            let _ = f.flush();
        }
        Err(e) => {
            logger::error!("Failed to open PID file: {}", e);
        }
    }
}

/// Removes the PID file at shutdown.
pub fn remove_pid_file() {
    let _ = fs::remove_file(paths::pid_file_path());
}
