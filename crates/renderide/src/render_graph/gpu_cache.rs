//! Small GPU cache primitives for render-graph effect passes.

use std::hash::Hash;
use std::sync::{Arc, OnceLock};

use hashbrown::HashMap;
use parking_lot::Mutex;

/// One-time GPU object slot.
#[derive(Debug)]
pub(crate) struct OnceGpu<T> {
    /// Lazily initialized GPU object.
    slot: OnceLock<T>,
}

impl<T> Default for OnceGpu<T> {
    fn default() -> Self {
        Self {
            slot: OnceLock::new(),
        }
    }
}

impl<T> OnceGpu<T> {
    /// Returns the cached object, creating it with `build` on first use.
    pub(crate) fn get_or_create(&self, build: impl FnOnce() -> T) -> &T {
        self.slot.get_or_init(build)
    }
}

/// Generic locked cache with double-check insertion and optional clear-on-overflow eviction.
#[derive(Debug)]
struct GpuCacheMap<K, V> {
    /// Cached objects keyed by pass-specific descriptors.
    entries: Mutex<HashMap<K, V>>,
    /// Maximum number of entries retained before the map is cleared.
    max_entries: Option<usize>,
}

impl<K, V> Default for GpuCacheMap<K, V> {
    fn default() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            max_entries: None,
        }
    }
}

impl<K, V> GpuCacheMap<K, V> {
    /// Creates an empty unbounded map.
    fn new() -> Self {
        Self::default()
    }

    /// Creates an empty map that clears itself before inserting once it reaches `max_entries`.
    fn with_max_entries(max_entries: usize) -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            max_entries: Some(max_entries),
        }
    }
}

impl<K, V> GpuCacheMap<K, V>
where
    K: Clone + Eq + Hash,
    V: Clone,
{
    /// Returns a cached value or builds, double-checks, and inserts a new one.
    fn get_or_create(&self, key: K, build: impl FnOnce(&K) -> V) -> V {
        {
            let guard = self.entries.lock();
            if let Some(existing) = guard.get(&key) {
                return existing.clone();
            }
        }

        let value = build(&key);
        let mut guard = self.entries.lock();
        if let Some(existing) = guard.get(&key) {
            return existing.clone();
        }
        if self
            .max_entries
            .is_some_and(|max_entries| guard.len() >= max_entries)
        {
            guard.clear();
        }
        guard.insert(key, value.clone());
        value
    }

    /// Clears all cached entries.
    #[cfg(test)]
    fn clear(&self) {
        self.entries.lock().clear();
    }

    /// Returns the number of cached entries.
    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.lock().len()
    }
}

/// Typed cache for `wgpu::RenderPipeline` values.
#[derive(Debug)]
pub(crate) struct RenderPipelineMap<K> {
    /// Shared map storing pipelines behind `Arc` so record paths can clone handles cheaply.
    inner: GpuCacheMap<K, Arc<wgpu::RenderPipeline>>,
}

impl<K> Default for RenderPipelineMap<K> {
    fn default() -> Self {
        Self {
            inner: GpuCacheMap::new(),
        }
    }
}

impl<K> RenderPipelineMap<K>
where
    K: Clone + Eq + Hash,
{
    /// Returns a cached render pipeline or builds one outside the map lock.
    pub(crate) fn get_or_create(
        &self,
        key: K,
        build: impl FnOnce(&K) -> wgpu::RenderPipeline,
    ) -> Arc<wgpu::RenderPipeline> {
        self.inner.get_or_create(key, |key| Arc::new(build(key)))
    }
}

/// Typed cache for `wgpu::BindGroup` values.
#[derive(Debug)]
pub(crate) struct BindGroupMap<K> {
    /// Shared map storing bind groups keyed by pass-specific resource identity.
    inner: GpuCacheMap<K, wgpu::BindGroup>,
}

impl<K> Default for BindGroupMap<K> {
    fn default() -> Self {
        Self {
            inner: GpuCacheMap::new(),
        }
    }
}

impl<K> BindGroupMap<K>
where
    K: Clone + Eq + Hash,
{
    /// Creates an empty bind-group map with clear-on-overflow eviction.
    pub(crate) fn with_max_entries(max_entries: usize) -> Self {
        Self {
            inner: GpuCacheMap::with_max_entries(max_entries),
        }
    }

    /// Returns a cached bind group or builds one outside the map lock.
    pub(crate) fn get_or_create(
        &self,
        key: K,
        build: impl FnOnce(&K) -> wgpu::BindGroup,
    ) -> wgpu::BindGroup {
        self.inner.get_or_create(key, build)
    }
}

#[cfg(test)]
mod tests {
    use super::GpuCacheMap;

    #[test]
    fn cache_separates_keys_and_reuses_values() {
        let cache = GpuCacheMap::<u32, u32>::new();

        assert_eq!(cache.get_or_create(1, |_| 10), 10);
        assert_eq!(cache.get_or_create(1, |_| 20), 10);
        assert_eq!(cache.get_or_create(2, |_| 20), 20);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn bounded_cache_clears_before_overflow_insert() {
        let cache = GpuCacheMap::<u32, u32>::with_max_entries(2);

        assert_eq!(cache.get_or_create(1, |_| 10), 10);
        assert_eq!(cache.get_or_create(2, |_| 20), 20);
        assert_eq!(cache.len(), 2);

        assert_eq!(cache.get_or_create(3, |_| 30), 30);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get_or_create(1, |_| 40), 40);
    }

    #[test]
    fn cache_clear_drops_entries() {
        let cache = GpuCacheMap::<u32, u32>::new();

        cache.get_or_create(1, |_| 10);
        cache.clear();

        assert_eq!(cache.len(), 0);
    }
}
