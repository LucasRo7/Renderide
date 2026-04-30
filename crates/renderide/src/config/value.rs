//! Validating value algebra for renderer config: bounded numeric ranges with optional zero-as-default
//! sentinels, plus power-of-two flooring for bloom-style mip pyramids.
//!
//! ## Why
//!
//! Several config fields kept their raw user-supplied integer in the section struct, then
//! re-validated it on every read through a hand-written helper (clamp to `[MIN, MAX]`, treat `0`
//! as "unset", round down to a power of two for graph use). The helpers were structurally
//! identical and scattered across [`super::types::rendering`] and [`super::types::post_processing`].
//!
//! [`Clamped`] consolidates the clamp-then-extract step into one type-driven primitive, and
//! [`power_of_two_floor`] consolidates the bloom-style rounding step into one place. Field types
//! stay as raw `u32` (so `config.toml` keeps loading numeric literals), but the resolver methods
//! now return [`Clamped`] / use [`power_of_two_floor`] internally rather than reimplementing the
//! arithmetic.

use std::fmt;

/// A `u32` known to satisfy `MIN <= value <= MAX`. Construct via [`Clamped::new`] (clamps an
/// arbitrary input) or [`Clamped::with_default_for_zero`] (treats `0` as a sentinel meaning
/// "use the supplied default").
///
/// `MIN` and `MAX` are const generics so the bounds are visible in error messages and the
/// returned value is structurally distinct from a plain `u32` at type-checking time. Callers
/// extract the raw value with [`Clamped::get`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Clamped<const MIN: u32, const MAX: u32>(u32);

impl<const MIN: u32, const MAX: u32> Clamped<MIN, MAX> {
    /// Smallest value this [`Clamped`] can hold, exposed as an associated constant for callers
    /// that want to log or display the bound.
    pub const MIN: u32 = MIN;
    /// Largest value this [`Clamped`] can hold.
    pub const MAX: u32 = MAX;

    /// Clamps `raw` into `[MIN, MAX]`.
    pub const fn new(raw: u32) -> Self {
        let v = if raw < MIN {
            MIN
        } else if raw > MAX {
            MAX
        } else {
            raw
        };
        Self(v)
    }

    /// Like [`Self::new`] but treats `raw == 0` as a sentinel meaning "use `default`". Several
    /// renderer config fields use `0` to mean "auto" / "unset"; the explicit fallback keeps a
    /// stray zero from being silently promoted to `MIN`.
    pub const fn with_default_for_zero(raw: u32, default: u32) -> Self {
        if raw == 0 {
            Self::new(default)
        } else {
            Self::new(raw)
        }
    }

    /// Returns the underlying clamped value.
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl<const MIN: u32, const MAX: u32> fmt::Display for Clamped<MIN, MAX> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<const MIN: u32, const MAX: u32> From<Clamped<MIN, MAX>> for u32 {
    fn from(c: Clamped<MIN, MAX>) -> Self {
        c.get()
    }
}

/// Rounds a non-zero `u32` down to the nearest power of two (`0` and `1` both map to `1`).
///
/// Used by bloom mip-pyramid sizing: arbitrary continuous dimensions in `config.toml` and the
/// HUD slider get rounded to a power of two before the graph builds the pyramid so every
/// downsample rung is stable.
pub const fn power_of_two_floor(value: u32) -> u32 {
    let v = if value == 0 { 1 } else { value };
    1_u32 << (u32::BITS - v.leading_zeros() - 1)
}

#[cfg(test)]
mod tests {
    use super::{Clamped, power_of_two_floor};

    #[test]
    fn clamped_clamps_into_range() {
        type C = Clamped<1, 3>;
        assert_eq!(C::new(0).get(), 1);
        assert_eq!(C::new(1).get(), 1);
        assert_eq!(C::new(2).get(), 2);
        assert_eq!(C::new(3).get(), 3);
        assert_eq!(C::new(99).get(), 3);
    }

    #[test]
    fn clamped_with_default_for_zero_uses_default_when_zero() {
        type C = Clamped<1, 3>;
        assert_eq!(C::with_default_for_zero(0, 2).get(), 2);
        assert_eq!(C::with_default_for_zero(1, 2).get(), 1);
        assert_eq!(C::with_default_for_zero(99, 2).get(), 3);
        // The default itself is also clamped, so a stray default outside the range still resolves safely.
        assert_eq!(C::with_default_for_zero(0, 99).get(), 3);
    }

    #[test]
    fn power_of_two_floor_rounds_correctly() {
        assert_eq!(power_of_two_floor(0), 1);
        assert_eq!(power_of_two_floor(1), 1);
        assert_eq!(power_of_two_floor(2), 2);
        assert_eq!(power_of_two_floor(3), 2);
        assert_eq!(power_of_two_floor(4), 4);
        assert_eq!(power_of_two_floor(7), 4);
        assert_eq!(power_of_two_floor(64), 64);
        assert_eq!(power_of_two_floor(65), 64);
        assert_eq!(power_of_two_floor(127), 64);
        assert_eq!(power_of_two_floor(128), 128);
        assert_eq!(power_of_two_floor(2048), 2048);
        assert_eq!(power_of_two_floor(2049), 2048);
    }

    #[test]
    fn clamped_into_u32_extracts_value() {
        type C = Clamped<10, 20>;
        let c = C::new(15);
        let raw: u32 = c.into();
        assert_eq!(raw, 15);
    }
}
