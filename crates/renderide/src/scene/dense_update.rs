//! Helpers for negative-terminated dense renderable update slabs.

/// Iterates non-negative entries until the host terminator.
pub(crate) fn non_negative_i32s(values: &[i32]) -> impl Iterator<Item = i32> + '_ {
    values.iter().copied().take_while(|&value| value >= 0)
}

/// Applies host dense-index removals with `swap_remove` semantics.
pub(crate) fn swap_remove_dense_indices<T>(rows: &mut Vec<T>, removals: &[i32]) {
    for raw in non_negative_i32s(removals) {
        let idx = raw as usize;
        if idx < rows.len() {
            rows.swap_remove(idx);
        }
    }
}

/// Pushes one row for each non-negative host addition id.
pub(crate) fn push_dense_additions<T>(
    rows: &mut Vec<T>,
    additions: &[i32],
    mut build: impl FnMut(i32) -> T,
) {
    for id in non_negative_i32s(additions) {
        rows.push(build(id));
    }
}

/// Removes transform ids invalidated by a dense transform removal.
pub(crate) fn retain_live_transform_ids(ids: &mut Vec<i32>) {
    ids.retain(|&id| id >= 0);
}

#[cfg(test)]
mod tests {
    use super::{push_dense_additions, swap_remove_dense_indices};

    #[test]
    fn removals_stop_at_negative_terminator() {
        let mut rows = vec![10, 20, 30];

        swap_remove_dense_indices(&mut rows, &[1, -1, 0]);

        assert_eq!(rows, vec![10, 30]);
    }

    #[test]
    fn additions_stop_at_negative_terminator() {
        let mut rows = vec![1];

        push_dense_additions(&mut rows, &[2, 3, -1, 4], |id| id * 10);

        assert_eq!(rows, vec![1, 20, 30]);
    }
}
