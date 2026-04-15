//! Host/renderer init handshake phase ([`InitState`]).

/// Host init sequence state (replaces paired booleans such as `init_received` / `init_finalized`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum InitState {
    /// Waiting for [`crate::shared::RendererCommand::RendererInitData`].
    #[default]
    Uninitialized,
    /// `renderer_init_data` received; waiting for [`crate::shared::RendererCommand::RendererInitFinalizeData`].
    InitReceived,
    /// Normal operation (or standalone mode).
    Finalized,
}

impl InitState {
    /// Whether host init handshake is complete.
    pub fn is_finalized(self) -> bool {
        matches!(self, InitState::Finalized)
    }
}
