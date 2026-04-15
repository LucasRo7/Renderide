//! Log severity ordering and `-LogLevel` argument parsing.

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
    /// Parses a level string (case-insensitive). Returns [`None`] for invalid values.
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

/// Stable `0..=4` tag for [`LogLevel`] (matches [`PartialOrd`] order).
#[inline]
pub(crate) fn level_to_tag(level: LogLevel) -> u8 {
    match level {
        LogLevel::Error => 0,
        LogLevel::Warn => 1,
        LogLevel::Info => 2,
        LogLevel::Debug => 3,
        LogLevel::Trace => 4,
    }
}

/// Maps a stored `0..=4` tag back to [`LogLevel`]. Values above `4` clamp to [`LogLevel::Trace`].
#[inline]
pub(crate) fn tag_to_level(tag: u8) -> LogLevel {
    match tag.min(4) {
        0 => LogLevel::Error,
        1 => LogLevel::Warn,
        2 => LogLevel::Info,
        3 => LogLevel::Debug,
        _ => LogLevel::Trace,
    }
}

/// Scans `exe` then args for a case-insensitive `-LogLevel` flag followed by a level value.
fn parse_loglevel_from_string_iter<I>(iter: I) -> Option<LogLevel>
where
    I: Iterator<Item = String>,
{
    let mut it = iter;
    while let Some(arg) = it.next() {
        if arg.eq_ignore_ascii_case("-LogLevel") {
            return it.next().and_then(|s| LogLevel::parse(&s));
        }
    }
    None
}

/// Parses `-LogLevel` from command line args (case-insensitive).
///
/// Returns [`None`] if not present or invalid; otherwise the parsed level.
///
/// Scans [`std::env::args`] without collecting argv into a [`Vec`].
pub fn parse_log_level_from_args() -> Option<LogLevel> {
    parse_loglevel_from_string_iter(std::env::args())
}

/// Roundtrip tests for [`level_to_tag`] and [`tag_to_level`], and `-LogLevel` argv parsing.
#[cfg(test)]
mod tag_tests {
    use super::*;

    fn tokens(args: &[&str]) -> Vec<String> {
        args.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn parse_log_level_from_slice_finds_flag() {
        assert_eq!(
            super::parse_loglevel_from_string_iter(
                tokens(&["prog", "-LogLevel", "debug"]).into_iter(),
            ),
            Some(LogLevel::Debug)
        );
    }

    #[test]
    fn parse_log_level_from_slice_case_insensitive_flag() {
        assert_eq!(
            super::parse_loglevel_from_string_iter(
                tokens(&["prog", "-loglevel", "INFO"]).into_iter(),
            ),
            Some(LogLevel::Info)
        );
    }

    #[test]
    fn parse_log_level_from_slice_ignores_other_tokens() {
        assert_eq!(
            super::parse_loglevel_from_string_iter(
                tokens(&["prog", "-x", "-LogLevel", "warn", "y"]).into_iter(),
            ),
            Some(LogLevel::Warn)
        );
    }

    #[test]
    fn parse_log_level_from_slice_missing_value() {
        assert!(
            super::parse_loglevel_from_string_iter(tokens(&["prog", "-LogLevel"]).into_iter(),)
                .is_none()
        );
    }

    #[test]
    fn parse_log_level_from_slice_absent() {
        assert!(
            super::parse_loglevel_from_string_iter(tokens(&["prog", "a", "b"]).into_iter())
                .is_none()
        );
    }

    #[test]
    fn level_tag_roundtrip() {
        for l in [
            LogLevel::Error,
            LogLevel::Warn,
            LogLevel::Info,
            LogLevel::Debug,
            LogLevel::Trace,
        ] {
            assert_eq!(tag_to_level(level_to_tag(l)), l);
        }
    }
}
