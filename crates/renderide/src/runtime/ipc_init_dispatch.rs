//! IPC command routing by [`crate::frontend::InitState`]: init handshake vs running dispatch.

use crate::frontend::InitState;
use crate::runtime::RendererRuntime;
use crate::shared::RendererCommand;

/// Dispatches a single command according to the current init phase.
pub(crate) fn dispatch_ipc_command(runtime: &mut RendererRuntime, cmd: RendererCommand) {
    match runtime.frontend.init_state() {
        InitState::Uninitialized => match cmd {
            RendererCommand::keep_alive(_) => {}
            RendererCommand::renderer_init_data(d) => runtime.on_init_data(d),
            _ => {
                logger::error!("IPC: expected RendererInitData first");
                runtime.frontend.fatal_error = true;
            }
        },
        InitState::InitReceived => match cmd {
            RendererCommand::keep_alive(_) => {}
            RendererCommand::renderer_init_finalize_data(_) => {
                runtime.frontend.set_init_state(InitState::Finalized);
            }
            RendererCommand::renderer_init_progress_update(_) => {}
            RendererCommand::renderer_engine_ready(_) => {}
            _ => {
                logger::trace!("IPC: deferring command until init finalized (skeleton)");
            }
        },
        InitState::Finalized => runtime.handle_running_command(cmd),
    }
}
