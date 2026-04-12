# Parallel cull / draw prep (future design)

Single-threaded [`collect_and_sort_world_mesh_draws`](crates/renderide/src/render_graph/world_mesh_draw_prep.rs)
and culling in [`world_mesh_cull`](crates/renderide/src/render_graph/world_mesh_cull.rs) are sufficient
today; this note records how to parallelize when profiling shows draw prep as a bottleneck.

## Data parallelism

- **Unit of work** — One [`RenderSpaceId`](crates/renderide/src/scene/mod.rs) (or batch of spaces)
  at a time: each has independent mesh renderable lists and sort keys.
- **Mechanism** — [`rayon`](https://crates.io/crates/rayon) `par_iter` over spaces for cull + sort,
  then deterministic merge if a global order is required; or a job queue with fixed worker count
  pinned to avoid oversubscription on low-core hosts.
- **Determinism** — Preserve material / sorting-order semantics by merging pre-sorted per-space
  streams with a k-way merge when the pipeline requires a single global draw list.

## Integration points

- CPU cull output feeds [`WorldMeshForwardPass`](crates/renderide/src/render_graph/passes/world_mesh_forward/mod.rs);
  keep scratch buffers per space or use a pool allocator to avoid cross-thread allocator contention.

This remains **design-only** until frame-time budgets justify the complexity and testing surface.
