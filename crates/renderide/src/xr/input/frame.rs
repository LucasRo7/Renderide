//! Per-frame resolved controller pose (grip/aim) before IPC mapping.

use glam::{Quat, Vec3};

use crate::shared::Chirality;

use super::pose::{bound_hand_pose_defaults, controller_pose_from_aim, touch_pose_correction};
use super::profile::ActiveControllerProfile;

/// Resolved controller and optional bound-hand pose in tracking space.
#[derive(Clone, Copy)]
pub(super) struct ControllerFrame {
    /// Controller position in host-tracking space.
    pub(super) position: Vec3,
    /// Controller orientation in host-tracking space.
    pub(super) rotation: Quat,
    /// Whether the frame carries a calibrated bound-hand offset.
    pub(super) has_bound_hand: bool,
    /// Bound-hand position relative to the controller pose.
    pub(super) hand_position: Vec3,
    /// Bound-hand rotation relative to the controller pose.
    pub(super) hand_rotation: Quat,
}

/// Touch-only: apply the `SteamVRDriver.UpdateController` Touch grip correction, else fall back
/// to an aim-derived pose.
fn resolve_touch_controller_frame(
    side: Chirality,
    grip_pose: Option<(Vec3, Quat)>,
    aim_pose: Option<(Vec3, Quat)>,
    has_bound_hand: bool,
    hand_position_default: Vec3,
    hand_rotation_default: Quat,
) -> Option<ControllerFrame> {
    if let Some((grip_position, grip_rotation)) = grip_pose {
        let (position, rotation) = touch_pose_correction(side, grip_position, grip_rotation);
        Some(ControllerFrame {
            position,
            rotation,
            has_bound_hand,
            hand_position: hand_position_default,
            hand_rotation: hand_rotation_default,
        })
    } else if let Some((aim_position, aim_rotation)) = aim_pose {
        let (position, rotation) = controller_pose_from_aim(aim_position, aim_rotation);
        Some(ControllerFrame {
            position,
            rotation,
            has_bound_hand,
            hand_position: hand_position_default,
            hand_rotation: hand_rotation_default,
        })
    } else {
        None
    }
}

/// All non-Touch profiles: use the raw grip pose directly (matching `SteamVRDriver`, which
/// applies no grip correction for Index / Vive / WMR / HP Reverb / Cosmos / Pico / Generic),
/// else fall back to an aim-derived pose.
fn resolve_generic_controller_frame(
    grip_pose: Option<(Vec3, Quat)>,
    aim_pose: Option<(Vec3, Quat)>,
    has_bound_hand: bool,
    hand_position_default: Vec3,
    hand_rotation_default: Quat,
) -> Option<ControllerFrame> {
    if let Some((grip_position, grip_rotation)) = grip_pose {
        Some(ControllerFrame {
            position: grip_position,
            rotation: grip_rotation,
            has_bound_hand,
            hand_position: hand_position_default,
            hand_rotation: hand_rotation_default,
        })
    } else if let Some((aim_position, aim_rotation)) = aim_pose {
        let (position, rotation) = controller_pose_from_aim(aim_position, aim_rotation);
        Some(ControllerFrame {
            position,
            rotation,
            has_bound_hand,
            hand_position: hand_position_default,
            hand_rotation: hand_rotation_default,
        })
    } else {
        None
    }
}

/// Per-profile pose resolution. Only Oculus Touch routes through
/// [`resolve_touch_controller_frame`]; every other profile uses the raw grip pose so the Rust
/// renderer matches `SteamVRDriver.UpdateController`, which applies no per-device grip correction
/// outside the Touch path.
pub(super) fn resolve_controller_frame(
    profile: ActiveControllerProfile,
    side: Chirality,
    grip_pose: Option<(Vec3, Quat)>,
    aim_pose: Option<(Vec3, Quat)>,
) -> Option<ControllerFrame> {
    let (has_bound_hand, hand_position_default, hand_rotation_default) =
        bound_hand_pose_defaults(profile, side);
    match profile {
        ActiveControllerProfile::Touch => resolve_touch_controller_frame(
            side,
            grip_pose,
            aim_pose,
            has_bound_hand,
            hand_position_default,
            hand_rotation_default,
        ),
        _ => resolve_generic_controller_frame(
            grip_pose,
            aim_pose,
            has_bound_hand,
            hand_position_default,
            hand_rotation_default,
        ),
    }
}

#[cfg(test)]
mod tests {
    use glam::{Quat, Vec3};

    use crate::shared::Chirality;

    use super::super::pose::{controller_pose_from_aim, touch_pose_correction};
    use super::super::profile::ActiveControllerProfile;
    use super::resolve_controller_frame;

    fn assert_vec3_near(actual: Vec3, expected: Vec3) {
        let delta = (actual - expected).length();
        assert!(
            delta < 1e-4,
            "vec3 mismatch: actual={actual:?} expected={expected:?} delta={delta}"
        );
    }

    fn assert_quat_near(actual: Quat, expected: Quat) {
        let dot = actual.normalize().dot(expected.normalize()).abs();
        assert!(
            (1.0 - dot) < 1e-4,
            "quat mismatch: actual={actual:?} expected={expected:?} dot={dot}"
        );
    }

    #[test]
    fn index_uses_grip_directly_with_identity_bound_hand() {
        let grip_position = Vec3::new(0.2, 1.3, -0.4);
        let grip_rotation = (Quat::from_rotation_y(0.6) * Quat::from_rotation_x(-0.2)).normalize();
        let aim_position = Vec3::new(0.24, 1.34, -0.28);
        let aim_rotation = (Quat::from_rotation_y(0.75) * Quat::from_rotation_x(-0.1)).normalize();

        let frame = resolve_controller_frame(
            ActiveControllerProfile::Index,
            Chirality::Left,
            Some((grip_position, grip_rotation)),
            Some((aim_position, aim_rotation)),
        )
        .expect("frame");

        assert_vec3_near(frame.position, grip_position);
        assert_quat_near(frame.rotation, grip_rotation);
        assert!(frame.has_bound_hand);
        assert_vec3_near(frame.hand_position, Vec3::ZERO);
        assert_quat_near(frame.hand_rotation, Quat::IDENTITY);
    }

    #[test]
    fn index_aim_only_matches_controller_pose_from_aim() {
        let aim_position = Vec3::new(0.24, 1.34, -0.28);
        let aim_rotation = (Quat::from_rotation_y(0.75) * Quat::from_rotation_x(-0.1)).normalize();
        let frame = resolve_controller_frame(
            ActiveControllerProfile::Index,
            Chirality::Left,
            None,
            Some((aim_position, aim_rotation)),
        )
        .expect("frame");
        let (expected_controller_position, expected_controller_rotation) =
            controller_pose_from_aim(aim_position, aim_rotation);
        assert_vec3_near(frame.position, expected_controller_position);
        assert_quat_near(frame.rotation, expected_controller_rotation);
    }

    #[test]
    fn generic_uses_aim_when_grip_missing() {
        let aim_position = Vec3::new(0.1, 1.2, -0.3);
        let aim_rotation = Quat::from_rotation_x(0.3);
        let frame = resolve_controller_frame(
            ActiveControllerProfile::Generic,
            Chirality::Right,
            None,
            Some((aim_position, aim_rotation)),
        )
        .expect("frame");
        let (expected_controller_position, expected_controller_rotation) =
            controller_pose_from_aim(aim_position, aim_rotation);
        assert_vec3_near(frame.position, expected_controller_position);
        assert_quat_near(frame.rotation, expected_controller_rotation);
    }

    #[test]
    fn touch_uses_aim_when_grip_missing() {
        let aim_position = Vec3::new(-0.2, 1.1, -0.25);
        let aim_rotation = Quat::from_rotation_y(-0.4);
        let frame = resolve_controller_frame(
            ActiveControllerProfile::Touch,
            Chirality::Left,
            None,
            Some((aim_position, aim_rotation)),
        )
        .expect("frame");
        let (expected_controller_position, expected_controller_rotation) =
            controller_pose_from_aim(aim_position, aim_rotation);
        assert_vec3_near(frame.position, expected_controller_position);
        assert_quat_near(frame.rotation, expected_controller_rotation);
    }

    #[test]
    fn touch_prefers_grip_when_both_present() {
        let grip_position = Vec3::new(0.2, 1.3, -0.4);
        let grip_rotation = (Quat::from_rotation_y(0.6) * Quat::from_rotation_x(-0.2)).normalize();
        let aim_position = Vec3::new(0.5, 0.5, 0.5);
        let aim_rotation = Quat::IDENTITY;
        let frame = resolve_controller_frame(
            ActiveControllerProfile::Touch,
            Chirality::Left,
            Some((grip_position, grip_rotation)),
            Some((aim_position, aim_rotation)),
        )
        .expect("frame");
        let (expected_pos, expected_rot) =
            touch_pose_correction(Chirality::Left, grip_position, grip_rotation);
        assert_vec3_near(frame.position, expected_pos);
        assert_quat_near(frame.rotation, expected_rot);
    }

    /// Pico / Reverb / WMR / Vive / Cosmos / Focus3 / Generic / Simple / Index all go through
    /// the generic path: grip is used directly with no per-device correction.
    #[test]
    fn non_touch_profiles_use_grip_directly() {
        let grip_position = Vec3::new(0.3, 1.2, -0.5);
        let grip_rotation = Quat::from_rotation_x(0.25).normalize();
        for profile in [
            ActiveControllerProfile::Index,
            ActiveControllerProfile::Vive,
            ActiveControllerProfile::WindowsMr,
            ActiveControllerProfile::HpReverbG2,
            ActiveControllerProfile::Pico4,
            ActiveControllerProfile::PicoNeo3,
            ActiveControllerProfile::ViveCosmos,
            ActiveControllerProfile::ViveFocus3,
            ActiveControllerProfile::Generic,
            ActiveControllerProfile::Simple,
        ] {
            let frame = resolve_controller_frame(
                profile,
                Chirality::Right,
                Some((grip_position, grip_rotation)),
                None,
            )
            .unwrap_or_else(|| panic!("frame for {profile:?}"));
            assert_vec3_near(frame.position, grip_position);
            assert_quat_near(frame.rotation, grip_rotation);
        }
    }
}
