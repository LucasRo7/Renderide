//! Renderer binary entry point.

fn main() {
    match renderide::run() {
        Ok(Some(code)) => std::process::exit(code),
        Ok(None) => {}
        Err(e) => {
            logger::error!("{e}");
            std::process::exit(1);
        }
    }
}
