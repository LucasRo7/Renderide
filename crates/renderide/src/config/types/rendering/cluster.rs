//! Clustered-light assignment backend selector.

use crate::labeled_enum;

labeled_enum! {
    /// Clustered-light assignment backend for the per-view light list pass.
    pub enum ClusterAssignmentMode: "cluster assignment mode" {
        default => Auto;

        /// Select the safest fast path for the current view. Currently uses CPU froxel
        /// assignment for the first stereo view only when the scene has many lights, otherwise
        /// falls back to GPU scan.
        Auto => {
            persist: "auto",
            label: "Auto",
        },
        /// Preserve the original compute shader path: one compute thread per froxel scans all
        /// lights.
        GpuScan => {
            persist: "gpu_scan",
            label: "GPU scan",
        },
        /// Use CPU light-centric froxel assignment where the shared cluster-buffer ordering
        /// permits it.
        CpuFroxel => {
            persist: "cpu_froxel",
            label: "CPU froxel",
        },
    }
}

#[cfg(test)]
mod tests {
    use super::ClusterAssignmentMode;
    use crate::config::types::RendererSettings;

    #[test]
    fn cluster_assignment_toml_roundtrip() {
        for mode in ClusterAssignmentMode::ALL.iter().copied() {
            let mut s = RendererSettings::default();
            s.rendering.cluster_assignment = mode;
            let toml = toml::to_string(&s).expect("serialize");
            let back: RendererSettings = toml::from_str(&toml).expect("deserialize");
            assert_eq!(back.rendering.cluster_assignment, mode);
        }
    }
}
