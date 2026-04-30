//! Material resolution and batch-key caching for world-mesh draw prep.

mod cache;
mod key;
mod keys;
mod resolve;

pub use cache::FrameMaterialBatchCache;
pub use key::{MaterialDrawBatchKey, compute_batch_key_hash};

pub(crate) use resolve::{MaterialResolveCtx, batch_key_for_slot_cached};
