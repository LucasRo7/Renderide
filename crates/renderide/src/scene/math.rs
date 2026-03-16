//! Math utilities for transform and matrix operations.

use nalgebra::{Matrix4, UnitQuaternion, Vector3};

use crate::shared::RenderTransform;

const MIN_SCALE: f32 = 1e-8;

/// Converts a RenderTransform to a 4x4 model matrix (translation * rotation * scale).
pub fn render_transform_to_matrix(t: &RenderTransform) -> Matrix4<f32> {
    let sx = if t.scale.x.is_finite() && t.scale.x.abs() >= MIN_SCALE {
        t.scale.x
    } else {
        1.0
    };
    let sy = if t.scale.y.is_finite() && t.scale.y.abs() >= MIN_SCALE {
        t.scale.y
    } else {
        1.0
    };
    let sz = if t.scale.z.is_finite() && t.scale.z.abs() >= MIN_SCALE {
        t.scale.z
    } else {
        1.0
    };
    let scale = Matrix4::new_nonuniform_scaling(&Vector3::new(sx, sy, sz));
    let rot: Matrix4<f32> = UnitQuaternion::try_new(t.rotation, 1e-8)
        .map(|u| u.to_homogeneous())
        .unwrap_or_else(Matrix4::identity);
    let pos = if t.position.x.is_finite() && t.position.y.is_finite() && t.position.z.is_finite() {
        t.position
    } else {
        Vector3::zeros()
    };
    let trans = Matrix4::new_translation(&pos);
    trans * rot * scale
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Quaternion;

    /// Round-trip test: position (1,2,3), identity rotation, scale (2,2,2).
    /// Verifies TRS order (trans * rot * scale) matches Unity Transform convention.
    #[test]
    fn test_render_transform_to_matrix_trs() {
        let t = RenderTransform {
            position: Vector3::new(1.0, 2.0, 3.0),
            scale: Vector3::new(2.0, 2.0, 2.0),
            rotation: Quaternion::identity(),
        };
        let m = render_transform_to_matrix(&t);
        let col3 = m.column(3);
        assert!((col3.x - 1.0).abs() < 1e-5, "translation x");
        assert!((col3.y - 2.0).abs() < 1e-5, "translation y");
        assert!((col3.z - 3.0).abs() < 1e-5, "translation z");
        assert!((m[(0, 0)] - 2.0).abs() < 1e-5, "scale xx");
        assert!((m[(1, 1)] - 2.0).abs() < 1e-5, "scale yy");
        assert!((m[(2, 2)] - 2.0).abs() < 1e-5, "scale zz");
    }
}
