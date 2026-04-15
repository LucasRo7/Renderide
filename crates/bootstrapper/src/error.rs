//! Errors returned by the bootstrapper `run` entry point.

use std::fmt;

/// Top-level failure from [`crate::run`].
#[derive(Debug)]
pub enum BootstrapError {
    /// Forwarded I/O error (filesystem, processes, etc.).
    Io(std::io::Error),
    /// Queue option or open failure with context.
    Interprocess(String),
    /// Logging could not be initialized.
    Logging(std::io::Error),
    /// Working directory could not be resolved.
    CurrentDir(std::io::Error),
    /// Shared-memory prefix could not be generated securely.
    Prefix(getrandom::Error),
}

impl fmt::Display for BootstrapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BootstrapError::Io(e) => write!(f, "{e}"),
            BootstrapError::Interprocess(s) => write!(f, "{s}"),
            BootstrapError::Logging(e) => write!(f, "logging: {e}"),
            BootstrapError::CurrentDir(e) => write!(f, "current directory: {e}"),
            BootstrapError::Prefix(e) => write!(f, "prefix generation: {e}"),
        }
    }
}

impl std::error::Error for BootstrapError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            BootstrapError::Io(e) => Some(e),
            BootstrapError::Logging(e) => Some(e),
            BootstrapError::CurrentDir(e) => Some(e),
            BootstrapError::Prefix(_) => None,
            BootstrapError::Interprocess(_) => None,
        }
    }
}

impl From<std::io::Error> for BootstrapError {
    fn from(value: std::io::Error) -> Self {
        BootstrapError::Io(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn display_io_forwards_message() {
        let inner = std::io::Error::new(std::io::ErrorKind::NotFound, "missing");
        let e = BootstrapError::Io(inner);
        assert_eq!(e.to_string(), "missing");
    }

    #[test]
    fn display_interprocess() {
        let e = BootstrapError::Interprocess("queue failed".to_string());
        assert_eq!(e.to_string(), "queue failed");
    }

    #[test]
    fn display_logging_prefix() {
        let inner = std::io::Error::other("disk full");
        let e = BootstrapError::Logging(inner);
        assert!(e.to_string().contains("logging"));
        assert!(e.to_string().contains("disk full"));
    }

    #[test]
    fn display_current_dir_prefix() {
        let inner = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let e = BootstrapError::CurrentDir(inner);
        assert!(e.to_string().contains("current directory"));
    }

    #[test]
    fn error_source_io_variants() {
        let io = std::io::Error::other("x");
        assert!(BootstrapError::Io(io).source().is_some());
        let io2 = std::io::Error::other("y");
        assert!(BootstrapError::Logging(io2).source().is_some());
    }
}
