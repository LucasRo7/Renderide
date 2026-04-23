//! Bootstrapper binary entry point.

#![warn(missing_docs)]
#![cfg_attr(windows, windows_subsystem = "windows")]

/// Parses CLI args, optionally prompts for desktop vs VR, then runs [`bootstrapper::run`].
///
/// Exits with status `0` without spawning the Host when the user cancels the
/// desktop vs VR dialog.
fn main() {
    let Some(opts) = bootstrapper::cli::prepare_run_inputs() else {
        return;
    };
    if let Err(e) = bootstrapper::run(opts) {
        logger::error!("{e}");
        std::process::exit(1);
    }
}
