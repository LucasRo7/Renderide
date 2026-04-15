//! Errors returned when opening queue backing storage or semaphores.

use std::io;

use thiserror::Error;

/// Error opening shared queue memory or creating the wakeup semaphore.
#[derive(Debug, Error)]
#[error(transparent)]
pub struct OpenError(#[from] pub io::Error);

/// Legacy alias used by earlier call sites.
pub type BackingError = OpenError;

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    #[test]
    fn open_error_display_forwards_io_message() {
        let inner = io::Error::new(io::ErrorKind::NotFound, "no mapping");
        let e = OpenError(inner);
        assert_eq!(e.to_string(), "no mapping");
    }
}
