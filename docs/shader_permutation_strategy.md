# Shader permutation strategy (design)

This document captures a **planning-only** direction for growing beyond a single
[`ShaderPermutation`](crates/renderide/src/pipelines/mod.rs) bitfield as features multiply (normal
mapping, shadow receiving, skinning, alpha test, etc.).

## Goals

- Avoid compiling every theoretical combination of toggles.
- Keep [`MaterialPipelineCacheKey`](crates/renderide/src/materials/cache.rs) stable and hashable.
- Allow lazy pipeline creation when a material + layout + permutation is first drawn.

## Proposed approach

1. **Feature key** — Either widen the bitfield with reserved ranges per domain (material / mesh /
   frame) or replace with a small struct of orthogonal enums/bitflags that implement `Hash` and
   `Eq` for the cache key.
2. **Lazy compilation** — On cache miss, compose WGSL (naga-oil) + build
   [`wgpu::RenderPipeline`]; optionally cap cache size with LRU eviction for unused variants.
3. **Pruning** — Ship only embedded stems that correspond to in-use Resonite/Unity families; drop
   unused keyword combinations from the default key space.
4. **Reflection alignment** — Keep permutation dimensions that affect `@group(1)` layout in sync
   with [`reflect_raster_material_wgsl`](crates/renderide/src/materials/wgsl_reflect.rs) so cache keys
   never disagree with bind layout.

No code in this area is required until a second orthogonal feature lands beside multiview stereo.
