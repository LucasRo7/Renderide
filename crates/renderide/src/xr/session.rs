//! OpenXR session frame loop: wait, begin, locate views, end.
//!
//! OpenXR [`xr::Posef`] is documented as transforming **from the space’s local frame to the
//! reference frame** (e.g. view pose in stage space). The camera **view** matrix is therefore
//! `inverse(T_ref_from_view)` — no coordinate basis conversion needed since the engine and
//! OpenXR stage space are both right-handed Y-up -Z-forward.
//!
//! ## Stereo convention (runtime `views` order)
//!
//! Per spec, `PRIMARY_STEREO` guarantees `views[0]` = left eye, `views[1]` = right eye.
//! [`headset_center_pose_from_stereo_views`] averages both for the center eye pose sent over IPC.
//! IPC uses [`openxr_pose_to_host_tracking`] (raw components, no transform).

use glam::{Mat4, Quat, Vec3};
use openxr as xr;
use openxr::{CompositionLayerProjection, CompositionLayerProjectionView, SwapchainSubImage};

use crate::render_graph::{apply_view_handedness_fix, reverse_z_perspective_openxr_fov};

/// `T_ref_from_view`: maps view-local points into the reference (stage) frame.
///
/// engine world space == OpenXR stage space (both RH, Y-up, -Z-forward), so no basis
/// conversion needed here. just raw pose -> mat4. the clip space Z-flip is handled
/// separately in apply_view_handedness_fix. -- xlinka 2026-04-07
#[inline]
pub(crate) fn ref_from_view_matrix(pose: &xr::Posef) -> Mat4 {
    let (translation, rotation) = openxr_pose_to_engine(pose);
    Mat4::from_rotation_translation(rotation, translation)
}

/// Per-eye view–projection from OpenXR [`xr::View`] (reverse-Z, engine handedness).
pub fn view_projection_from_xr_view(view: &xr::View, near: f32, far: f32) -> Mat4 {
    let ref_from_view = ref_from_view_matrix(&view.pose);
    let view_mat = apply_view_handedness_fix(ref_from_view.inverse());
    let proj = reverse_z_perspective_openxr_fov(&view.fov, near, far);
    proj * view_mat
}

/// Maps an OpenXR [`xr::Posef`] to **rendering** translation + rotation.
///
/// engine space and OpenXR stage space are the same (RH, Y-up, -Z-forward), so this is
/// just raw component extraction + normalization. previously this did a diag(-1,1,-1)
/// sandwich which was fucking up pitch and roll -- xlinka 2026-04-07
///
/// **Do not** use this for [`crate::shared::HeadsetState`] / FrooxEngine IPC; use
/// [`openxr_pose_to_host_tracking`] instead.
pub fn openxr_pose_to_engine(pose: &xr::Posef) -> (Vec3, Quat) {
    openxr_pose_to_host_tracking(pose)
}

/// Position and orientation for **host IPC** (FrooxEngine [`crate::shared::HeadsetState`]), matching
/// Unity’s XR tracking convention: **raw** OpenXR [`xr::Posef`] components as
/// [`glam::Vec3`] / [`glam::Quat`] (same idea as `Vector3.ToRender` / `Quaternion.ToRender` in the
/// Unity bridge — no reflection `S` sandwich). Use this for headset and controller poses sent to the
/// engine; use [`openxr_pose_to_engine`] only for GPU view–projection construction.
pub fn openxr_pose_to_host_tracking(pose: &xr::Posef) -> (Vec3, Quat) {
    let p = Vec3::new(pose.position.x, pose.position.y, pose.position.z);
    let o = pose.orientation;
    let q = Quat::from_xyzw(o.x, o.y, o.z, o.w);
    let len_sq = q.length_squared();
    let q = if len_sq.is_finite() && len_sq >= 1e-10 {
        q.normalize()
    } else {
        Quat::IDENTITY
    };
    (p, q)
}

/// Headset pose for IPC in host tracking space ([`openxr_pose_to_host_tracking`]).
pub fn headset_pose_from_xr_view(view: &xr::View) -> (Vec3, Quat) {
    openxr_pose_to_host_tracking(&view.pose)
}

/// Approximates **center eye** (Unity `XRNode.CenterEye`): averages per-eye positions and slerps
/// orientations from the first two stereo [`xr::View`] entries using [`openxr_pose_to_host_tracking`].
pub fn headset_center_pose_from_stereo_views(views: &[xr::View]) -> Option<(Vec3, Quat)> {
    match views.len() {
        0 => None,
        1 => Some(headset_pose_from_xr_view(&views[0])),
        _ => {
            let (p0, r0) = openxr_pose_to_host_tracking(&views[0].pose);
            let (p1, r1) = openxr_pose_to_host_tracking(&views[1].pose);
            let pos = (p0 + p1) * 0.5;
            let rot = r0.slerp(r1, 0.5).normalize();
            Some((pos, rot))
        }
    }
}

/// OpenXR requires a unit quaternion; some runtimes briefly report `(0,0,0,0)`, which makes
/// `xrEndFrame` fail with `XR_ERROR_POSE_INVALID`.
fn sanitize_pose_for_end_frame(pose: xr::Posef) -> xr::Posef {
    let o = pose.orientation;
    let len_sq = o.x * o.x + o.y * o.y + o.z * o.z + o.w * o.w;
    if len_sq.is_finite() && len_sq >= 1e-10 {
        pose
    } else {
        xr::Posef {
            orientation: xr::Quaternionf {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            },
            position: pose.position,
        }
    }
}

/// Owns OpenXR session objects (constructed in [`super::bootstrap::init_wgpu_openxr`]).
pub struct XrSessionState {
    pub(super) xr_instance: xr::Instance,
    pub(super) environment_blend_mode: xr::EnvironmentBlendMode,
    pub(super) session: xr::Session<xr::Vulkan>,
    pub(super) session_running: bool,
    pub(super) frame_wait: xr::FrameWaiter,
    pub(super) frame_stream: xr::FrameStream<xr::Vulkan>,
    pub(super) stage: xr::Space,
    pub(super) event_storage: xr::EventDataBuffer,
}

impl XrSessionState {
    pub(super) fn new(
        xr_instance: xr::Instance,
        environment_blend_mode: xr::EnvironmentBlendMode,
        session: xr::Session<xr::Vulkan>,
        frame_wait: xr::FrameWaiter,
        frame_stream: xr::FrameStream<xr::Vulkan>,
        stage: xr::Space,
    ) -> Self {
        Self {
            xr_instance,
            environment_blend_mode,
            session,
            session_running: false,
            frame_wait,
            frame_stream,
            stage,
            event_storage: xr::EventDataBuffer::new(),
        }
    }

    /// Poll events and return `false` if the session should exit.
    pub fn poll_events(&mut self) -> Result<bool, xr::sys::Result> {
        while let Some(event) = self.xr_instance.poll_event(&mut self.event_storage)? {
            use xr::Event::*;
            match event {
                SessionStateChanged(e) => match e.state() {
                    xr::SessionState::READY => {
                        self.session
                            .begin(xr::ViewConfigurationType::PRIMARY_STEREO)?;
                        self.session_running = true;
                    }
                    xr::SessionState::STOPPING => {
                        self.session.end()?;
                        self.session_running = false;
                    }
                    xr::SessionState::EXITING | xr::SessionState::LOSS_PENDING => {
                        return Ok(false);
                    }
                    _ => {}
                },
                InstanceLossPending(_) => return Ok(false),
                _ => {}
            }
        }
        Ok(true)
    }

    /// Whether the OpenXR session is running.
    pub fn session_running(&self) -> bool {
        self.session_running
    }

    /// OpenXR instance handle (swapchain creation, view enumeration).
    pub fn xr_instance(&self) -> &xr::Instance {
        &self.xr_instance
    }

    /// Underlying Vulkan session (swapchain lifetime).
    pub fn xr_vulkan_session(&self) -> &xr::Session<xr::Vulkan> {
        &self.session
    }

    /// Stage reference space used for [`Self::locate_views`] and controller [`xr::Space`] location.
    pub fn stage_space(&self) -> &xr::Space {
        &self.stage
    }

    /// Blocks until the next frame, begins the frame stream. Returns `None` if not ready or idle.
    pub fn wait_frame(&mut self) -> Result<Option<xr::FrameState>, xr::sys::Result> {
        if !self.session_running {
            std::thread::sleep(std::time::Duration::from_millis(10));
            return Ok(None);
        }
        let state = self.frame_wait.wait()?;
        self.frame_stream.begin()?;
        Ok(Some(state))
    }

    /// Ends the frame with no composition layers (mirror path until swapchain submission is wired).
    pub fn end_frame_empty(
        &mut self,
        predicted_display_time: xr::Time,
    ) -> Result<(), xr::sys::Result> {
        self.frame_stream
            .end(predicted_display_time, self.environment_blend_mode, &[])
    }

    /// Submits a stereo projection layer referencing the acquired swapchain image (`image_rect` in pixels).
    ///
    /// spec guarantees views[0] = left, views[1] = right for PRIMARY_STEREO.
    /// layer 0 / image_array_index 0 = left eye, layer 1 / index 1 = right eye.
    /// multiview view_index in shader matches this ordering. -- xlinka 2026-04-07
    pub fn end_frame_projection(
        &mut self,
        predicted_display_time: xr::Time,
        swapchain: &xr::Swapchain<xr::Vulkan>,
        views: &[xr::View],
        image_rect: xr::Rect2Di,
    ) -> Result<(), xr::sys::Result> {
        if views.len() < 2 {
            return self.end_frame_empty(predicted_display_time);
        }
        let v0 = &views[0]; // left eye
        let v1 = &views[1]; // right eye
        let pose0 = sanitize_pose_for_end_frame(v0.pose);
        let pose1 = sanitize_pose_for_end_frame(v1.pose);
        let projection_views = [
            CompositionLayerProjectionView::new()
                .pose(pose0)
                .fov(v0.fov)
                .sub_image(
                    SwapchainSubImage::new()
                        .swapchain(swapchain)
                        .image_array_index(0)
                        .image_rect(image_rect),
                ),
            CompositionLayerProjectionView::new()
                .pose(pose1)
                .fov(v1.fov)
                .sub_image(
                    SwapchainSubImage::new()
                        .swapchain(swapchain)
                        .image_array_index(1)
                        .image_rect(image_rect),
                ),
        ];
        let layer = CompositionLayerProjection::new()
            .space(&self.stage)
            .views(&projection_views);
        self.frame_stream.end(
            predicted_display_time,
            self.environment_blend_mode,
            &[&layer],
        )
    }

    /// Locates stereo views for the predicted display time.
    pub fn locate_views(
        &self,
        predicted_display_time: xr::Time,
    ) -> Result<Vec<xr::View>, xr::sys::Result> {
        let (_, views) = self.session.locate_views(
            xr::ViewConfigurationType::PRIMARY_STEREO,
            predicted_display_time,
            &self.stage,
        )?;
        Ok(views)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openxr as xr;

    fn pose_identity() -> xr::Posef {
        xr::Posef {
            orientation: xr::Quaternionf {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            },
            position: xr::Vector3f {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        }
    }

    #[test]
    fn identity_pose_maps_to_identity_ref_from_view() {
        let m = ref_from_view_matrix(&pose_identity());
        assert!(
            m.abs_diff_eq(Mat4::IDENTITY, 1e-4),
            "expected identity ref_from_view, got {m:?}"
        );
    }

    #[test]
    fn identity_openxr_pose_maps_to_identity_engine_quat() {
        let (_p, q) = openxr_pose_to_engine(&pose_identity());
        assert!(
            q.abs_diff_eq(Quat::IDENTITY, 1e-4),
            "expected identity engine orientation, got {q:?}"
        );
    }

    #[test]
    fn host_tracking_pose_matches_raw_openxr_components() {
        let pose = xr::Posef {
            orientation: xr::Quaternionf {
                x: 0.1,
                y: 0.2,
                z: 0.3,
                w: 0.9,
            },
            position: xr::Vector3f {
                x: 1.0,
                y: 2.0,
                z: -3.0,
            },
        };
        let (p, q) = openxr_pose_to_host_tracking(&pose);
        assert!(p.abs_diff_eq(Vec3::new(1.0, 2.0, -3.0), 1e-5));
        let o = pose.orientation;
        let q_raw = Quat::from_xyzw(o.x, o.y, o.z, o.w).normalize();
        assert!(q.abs_diff_eq(q_raw, 1e-4));
    }

    #[test]
    fn headset_center_pose_averages_positions_and_slerps_rotation() {
        let pose_l = xr::Posef {
            orientation: xr::Quaternionf {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            },
            position: xr::Vector3f {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        };
        let pose_r = xr::Posef {
            orientation: xr::Quaternionf {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            },
            position: xr::Vector3f {
                x: 0.2,
                y: 0.0,
                z: 0.0,
            },
        };
        let views = [
            xr::View {
                pose: pose_l,
                fov: xr::Fovf {
                    angle_left: 0.0,
                    angle_right: 0.0,
                    angle_up: 0.0,
                    angle_down: 0.0,
                },
            },
            xr::View {
                pose: pose_r,
                fov: xr::Fovf {
                    angle_left: 0.0,
                    angle_right: 0.0,
                    angle_up: 0.0,
                    angle_down: 0.0,
                },
            },
        ];
        let (p, q) = headset_center_pose_from_stereo_views(&views).expect("center pose");
        let (pl, _) = openxr_pose_to_host_tracking(&pose_l);
        let (pr, _) = openxr_pose_to_host_tracking(&pose_r);
        let expected_p = (pl + pr) * 0.5;
        assert!(
            p.abs_diff_eq(expected_p, 1e-4),
            "p={p:?} expected {expected_p:?}"
        );
        assert!(q.abs_diff_eq(Quat::IDENTITY, 1e-4));
    }

    #[test]
    fn pitch_up_forward_has_positive_y_in_stage() {
        // xlinka 2026-04-07: this is the test that was actually missing. the old tests
        // only checked openxr_pose_to_engine and ref_from_view_matrix agreed with each
        // other -- which they did, even when both were wrong. congrats to past me i guess.
        //
        // positive X rotation in OpenXR = HMD pitching upward (right-hand rule: thumb
        // along +X, the -Z forward axis tilts toward +Y). forward in stage space should
        // have a positive Y component. if this breaks, pitch is inverted again.
        let angle = 0.3_f32;
        let q_xr = Quat::from_rotation_x(angle);
        let pose = xr::Posef {
            orientation: xr::Quaternionf {
                x: q_xr.x,
                y: q_xr.y,
                z: q_xr.z,
                w: q_xr.w,
            },
            position: xr::Vector3f {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        };
        let ref_from_view = ref_from_view_matrix(&pose);
        let forward_in_stage = ref_from_view.transform_vector3(-Vec3::Z);
        assert!(
            forward_in_stage.y > 0.0,
            "pitch up should tilt forward toward +Y in stage, got {forward_in_stage:?}"
        );
    }

    #[test]
    fn yaw_right_forward_has_negative_x_in_stage() {
        // xlinka 2026-04-07: positive Y rotation = yaw right. in RH Y-up -Z-forward
        // space, rotating -Z toward -X means forward.x should go negative when yawing right.
        let angle = 0.3_f32;
        let q_xr = Quat::from_rotation_y(angle);
        let pose = xr::Posef {
            orientation: xr::Quaternionf {
                x: q_xr.x,
                y: q_xr.y,
                z: q_xr.z,
                w: q_xr.w,
            },
            position: xr::Vector3f {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        };
        let ref_from_view = ref_from_view_matrix(&pose);
        let forward_in_stage = ref_from_view.transform_vector3(-Vec3::Z);
        assert!(
            forward_in_stage.x < 0.0,
            "yaw right should tilt forward toward -X in stage, got {forward_in_stage:?}"
        );
    }
}
