//! Host transport and session layer: IPC queues, shared memory, init handshake, lock-step gating.
//!
//! Layout: [`renderer_frontend`] ([`RendererFrontend`]) composes small transport, session,
//! lock-step, performance, output-policy, and decoupling components; [`dispatch`] owns IPC command
//! classification/routing; [`input`] adapts winit/XR snapshots into [`crate::shared::InputState`].
//!
//! [`RendererFrontend`] is the side-effect facade for queue and shared-memory access. Pure
//! decisions such as begin-frame gating, init routing, output policy, and decoupling transitions
//! live in their domain modules and are applied by the facade/runtime.

mod begin_frame;
mod decoupling;
pub(crate) mod dispatch;
mod frame_start_performance;
mod init_state;
mod lockstep_state;
mod output_policy;
mod renderer_frontend;
mod session;
mod transport;

/// Winit adapter and [`WindowInputAccumulator`](input::WindowInputAccumulator) for [`crate::shared::InputState`].
pub mod input;

pub use decoupling::DecouplingState;
pub use init_state::InitState;
pub use renderer_frontend::RendererFrontend;
