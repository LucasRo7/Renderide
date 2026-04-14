//! Validates `[offset, offset+length)` against a mapped buffer’s total length.

/// Converts `offset`/`length` into a valid byte subrange of `total_len`, or `None`.
pub(super) fn byte_subrange(total_len: usize, offset: i32, length: i32) -> Option<(usize, usize)> {
    let offset = usize::try_from(offset).ok()?;
    let length = usize::try_from(length).ok()?;
    let end = offset.checked_add(length)?;
    if end <= total_len {
        Some((offset, end))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::byte_subrange;

    #[test]
    fn byte_subrange_ok_and_rejects_overflow() {
        assert_eq!(byte_subrange(100, 10, 5), Some((10, 15)));
        assert_eq!(byte_subrange(100, 0, 100), Some((0, 100)));
        assert_eq!(byte_subrange(100, 99, 2), None);
        assert_eq!(byte_subrange(100, -1, 5), None);
    }
}
