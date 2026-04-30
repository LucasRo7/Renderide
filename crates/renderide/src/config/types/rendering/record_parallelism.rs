//! Per-view encoder recording strategy (serial vs rayon-parallel).

use crate::labeled_enum;

labeled_enum! {
    /// Controls whether per-view encoder recording uses rayon for parallelism.
    ///
    /// The default [`RecordParallelism::PerViewParallel`] records per-view encoders on rayon
    /// workers for CPU-side speedup on stereo / multi-camera scenes. Switch to
    /// [`RecordParallelism::Serial`] only for debugging or when isolating regressions.
    pub enum RecordParallelism: "record parallelism" {
        default => PerViewParallel;

        /// Record each per-view encoder sequentially on the main thread. Safe and debuggable.
        Serial => {
            persist: "serial",
            label: "Serial",
        },
        /// Record each per-view encoder on a rayon worker thread. Requires all per-view pass
        /// nodes to be `Send` (enforced at compile time by the trait bound on
        /// [`crate::render_graph::PassNode`]).
        PerViewParallel => {
            persist: "per_view_parallel",
            label: "Per-view parallel",
        },
    }
}
