//! Errors returned by the bootstrapper `run` entry point.

use thiserror::Error;

/// Top-level failure from [`crate::run`].
#[derive(Debug, Error)]
pub enum BootstrapError {
    /// Forwarded I/O error (filesystem, processes, etc.).
    #[error("{0}")]
    Io(
        #[from]
        #[source]
        std::io::Error,
    ),
    /// Queue option or open failure with context.
    #[error("{0}")]
    Interprocess(String),
    /// Logging could not be initialized.
    #[error("logging: {0}")]
    Logging(#[source] std::io::Error),
    /// Working directory could not be resolved.
    #[error("current directory: {0}")]
    CurrentDir(#[source] std::io::Error),
    /// Shared-memory prefix could not be generated securely.
    #[error("prefix generation: {0}")]
    Prefix(#[source] getrandom::Error),
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
