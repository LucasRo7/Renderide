//! Tiered file logging for Renderide and bootstrapper.
//!
//! Provides `LogLevel`, `init()`, `log_panic()`, and macros `error!`, `warn!`, `info!`, `debug!`, `trace!`.
//! Call `init(path, max_level, append)` once at startup; use `log_panic(path, info)` in panic hooks.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;
use std::sync::OnceLock;

/// Log level for filtering. Lower ordinal = higher priority.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    /// Critical errors.
    Error,
    /// Warnings.
    Warn,
    /// Informational messages.
    Info,
    /// Debug diagnostics.
    Debug,
    /// Verbose trace.
    Trace,
}

impl LogLevel {
    /// Parses a level string (case-insensitive). Returns None for invalid values.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "error" | "e" => Some(Self::Error),
            "warn" | "warning" | "w" => Some(Self::Warn),
            "info" | "i" => Some(Self::Info),
            "debug" | "d" => Some(Self::Debug),
            "trace" | "t" => Some(Self::Trace),
            _ => None,
        }
    }

    /// Returns the string to pass as `-LogLevel` value.
    pub fn as_arg(&self) -> &'static str {
        match self {
            Self::Error => "error",
            Self::Warn => "warn",
            Self::Info => "info",
            Self::Debug => "debug",
            Self::Trace => "trace",
        }
    }
}

impl std::fmt::Debug for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Error => write!(f, "ERROR"),
            Self::Warn => write!(f, "WARN"),
            Self::Info => write!(f, "INFO"),
            Self::Debug => write!(f, "DEBUG"),
            Self::Trace => write!(f, "TRACE"),
        }
    }
}

/// Logger that writes to file and optionally to stderr.
struct Logger {
    /// File output. Mutex for thread-safe writes.
    file: Mutex<std::fs::File>,
    /// Whether to also write to stderr.
    console: bool,
    /// Maximum level to log. Messages at or below this level are written.
    max_level: LogLevel,
}

/// Global logger instance. Set by `init()`.
static LOGGER: OnceLock<Logger> = OnceLock::new();

/// Parses `-LogLevel` from command line args (case-insensitive).
/// Returns `None` if not present or invalid; otherwise the parsed level.
pub fn parse_log_level_from_args() -> Option<LogLevel> {
    let args: Vec<String> = std::env::args().collect();
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if arg.eq_ignore_ascii_case("-LogLevel") && i + 1 < args.len() {
            return LogLevel::parse(&args[i + 1]);
        }
        i += 1;
    }
    None
}

/// Initializes logging. Creates parent directory if needed, opens file.
/// Call once at startup before panic hook.
///
/// # Arguments
/// * `path` - Path to the log file.
/// * `max_level` - Maximum level to log (messages at or below this level are written).
/// * `append` - If true, append to file; if false, truncate.
pub fn init(path: impl AsRef<Path>, max_level: LogLevel, append: bool) {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut opts = OpenOptions::new();
    opts.create(true).write(true);
    if append {
        opts.append(true);
    } else {
        opts.truncate(true);
    }
    let file = match opts.open(path)
    {
        Ok(f) => f,
        Err(_) => return,
    };
    let logger = Logger {
        file: Mutex::new(file),
        console: false,
        max_level,
    };
    let _ = LOGGER.set(logger);
}

/// Flushes any buffered log output. For `std::fs::File`, `flush()` is a no-op (data goes
/// to the kernel on write). Call periodically for API consistency.
///
/// Do not call from a panic hook: if the panic occurred while holding the logger mutex
/// (e.g. inside a log macro), this would deadlock.
pub fn flush() {
    if let Some(logger) = LOGGER.get() {
        if let Ok(mut file) = logger.file.lock() {
            let _ = file.flush();
        }
    }
}

/// Writes panic info and backtrace to the log file. Flushes immediately so the panic
/// is visible on disk. Does not acquire the logger mutex (safe from panic handler).
pub fn log_panic(path: impl AsRef<Path>, info: &dyn std::fmt::Display) {
    let path = path.as_ref();
    if let Ok(mut f) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = writeln!(f, "PANIC: {}", info);
        let _ = writeln!(f, "Backtrace:\n{:?}", std::backtrace::Backtrace::capture());
        let _ = f.flush();
        let _ = f.sync_all();
    }
}

/// Internal log writer. Called by the log macros.
#[doc(hidden)]
pub fn log(level: LogLevel, args: std::fmt::Arguments<'_>) {
    if let Some(logger) = LOGGER.get() && level <= logger.max_level {
        let msg = args.to_string();
        let timestamp = format_timestamp();
        let line = format!("[{}] {:?} {}\n", timestamp, level, msg);
        if let Ok(mut file) = logger.file.lock() {
            let _ = file.write_all(line.as_bytes());
            let _ = file.flush();
        }
        if logger.console {
            let _ = std::io::stderr().write_all(line.as_bytes());
            let _ = std::io::stderr().flush();
        }
    }
}

/// Returns a simple timestamp string. Uses std::time for a minimal format.
fn format_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let Ok(dur) = SystemTime::now().duration_since(UNIX_EPOCH) else {
        return "?".to_string();
    };
    let secs = dur.as_secs();
    let millis = dur.subsec_millis();
    let mins = (secs / 60) % 60;
    let hours = (secs / 3600) % 24;
    let secs = secs % 60;
    format!("{:02}:{:02}:{:02}.{:03}", hours, mins, secs, millis)
}

/// Logs at error level.
#[macro_export]
macro_rules! error {
    ($($arg:tt)*) => {
        $crate::log($crate::LogLevel::Error, format_args!($($arg)*))
    };
}

/// Logs at warn level.
#[macro_export]
macro_rules! warn {
    ($($arg:tt)*) => {
        $crate::log($crate::LogLevel::Warn, format_args!($($arg)*))
    };
}

/// Logs at info level.
#[macro_export]
macro_rules! info {
    ($($arg:tt)*) => {
        $crate::log($crate::LogLevel::Info, format_args!($($arg)*))
    };
}

/// Logs at debug level.
#[macro_export]
macro_rules! debug {
    ($($arg:tt)*) => {
        $crate::log($crate::LogLevel::Debug, format_args!($($arg)*))
    };
}

/// Logs at trace level.
#[macro_export]
macro_rules! trace {
    ($($arg:tt)*) => {
        $crate::log($crate::LogLevel::Trace, format_args!($($arg)*))
    };
}
