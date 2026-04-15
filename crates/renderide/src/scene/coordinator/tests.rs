//! Unit tests for [`super::SceneCoordinator`].

use glam::{Mat4, Quat, Vec3};

use crate::render_graph::{
    view_matrix_for_world_mesh_render_space, view_matrix_from_render_transform,
};
use crate::scene::render_space::RenderSpaceState;
use crate::shared::RenderingContext;

use super::super::ids::RenderSpaceId;
use super::super::world::{compute_world_matrices_for_space, WorldTransformCache};
use super::SceneCoordinator;

impl SceneCoordinator {
    /// Inserts a render space and solves world matrices from the given locals (for unit tests).
    pub(crate) fn test_seed_space_identity_worlds(
        &mut self,
        id: RenderSpaceId,
        nodes: Vec<crate::shared::RenderTransform>,
        node_parents: Vec<i32>,
    ) {
        assert_eq!(
            nodes.len(),
            node_parents.len(),
            "nodes and node_parents length must match"
        );
        self.spaces.insert(
            id,
            RenderSpaceState {
                id,
                is_active: true,
                nodes,
                node_parents,
                ..Default::default()
            },
        );
        let space = self.spaces.get(&id).expect("inserted space");
        let mut cache = WorldTransformCache::default();
        let _ =
            compute_world_matrices_for_space(id.0, &space.nodes, &space.node_parents, &mut cache);
        self.world_caches.insert(id, cache);
    }
}

#[test]
fn world_matrix_excludes_render_space_root() {
    let mut scene = SceneCoordinator::new();
    let id = RenderSpaceId(1);
    scene.spaces.insert(
        id,
        RenderSpaceState {
            id,
            is_active: true,
            root_transform: crate::shared::RenderTransform {
                position: Vec3::new(100.0, 0.0, 0.0),
                scale: Vec3::ONE,
                rotation: Quat::IDENTITY,
            },
            nodes: vec![crate::shared::RenderTransform {
                position: Vec3::new(1.0, 2.0, 3.0),
                scale: Vec3::ONE,
                rotation: Quat::IDENTITY,
            }],
            node_parents: vec![-1],
            ..Default::default()
        },
    );
    let space = scene.spaces.get(&id).expect("space");
    let mut cache = WorldTransformCache::default();
    compute_world_matrices_for_space(id.0, &space.nodes, &space.node_parents, &mut cache)
        .expect("solve");
    scene.world_caches.insert(id, cache);

    let world = scene.world_matrix(id, 0).expect("matrix");
    let t = world.col(3);
    assert!(
        (t.x - 1.0).abs() < 1e-4,
        "world_matrix must not include root_transform translation (got x={})",
        t.x
    );

    let with_root = scene
        .world_matrix_including_space_root(id, 0)
        .expect("with root");
    let t2 = with_root.col(3);
    assert!(
        (t2.x - 101.0).abs() < 0.1,
        "world_matrix_including_space_root should add root translation (got x={})",
        t2.x
    );
}

#[test]
fn overlay_render_matrix_tracks_head_output_transform() {
    let mut scene = SceneCoordinator::new();
    let id = RenderSpaceId(7);
    scene.spaces.insert(
        id,
        RenderSpaceState {
            id,
            is_active: true,
            is_overlay: true,
            root_transform: crate::shared::RenderTransform {
                position: Vec3::new(2.0, 3.0, 4.0),
                scale: Vec3::ONE,
                rotation: Quat::IDENTITY,
            },
            nodes: vec![crate::shared::RenderTransform {
                position: Vec3::new(1.0, 0.0, 0.0),
                scale: Vec3::ONE,
                rotation: Quat::IDENTITY,
            }],
            node_parents: vec![-1],
            ..Default::default()
        },
    );
    let space = scene.spaces.get(&id).expect("space");
    let mut cache = WorldTransformCache::default();
    compute_world_matrices_for_space(id.0, &space.nodes, &space.node_parents, &mut cache)
        .expect("solve");
    scene.world_caches.insert(id, cache);

    let head_output =
        Mat4::from_scale_rotation_translation(Vec3::ONE, Quat::IDENTITY, Vec3::new(10.0, 0.0, 0.0));
    let world = scene
        .world_matrix_for_render_context(id, 0, RenderingContext::UserView, head_output)
        .expect("render matrix");
    let t = world.col(3);
    assert!(
        (t.x - 9.0).abs() < 1e-4,
        "overlay x should follow head output"
    );
    assert!(
        (t.y + 3.0).abs() < 1e-4,
        "overlay y should subtract space root"
    );
    assert!(
        (t.z + 4.0).abs() < 1e-4,
        "overlay z should subtract space root"
    );
}

/// Overlay spaces use the main camera view because object matrices are in main-world coordinates.
#[test]
fn overlay_render_space_view_matrix_matches_main_space() {
    let mut scene = SceneCoordinator::new();
    let main_id = RenderSpaceId(1);
    let overlay_id = RenderSpaceId(0);
    scene.spaces.insert(
        main_id,
        RenderSpaceState {
            id: main_id,
            is_active: true,
            is_overlay: false,
            override_view_position: true,
            root_transform: crate::shared::RenderTransform {
                position: Vec3::new(10.0, 0.0, 0.0),
                scale: Vec3::ONE,
                rotation: Quat::IDENTITY,
            },
            view_transform: crate::shared::RenderTransform {
                position: Vec3::new(10.0, 1.7, 5.0),
                scale: Vec3::ONE,
                rotation: Quat::IDENTITY,
            },
            ..Default::default()
        },
    );
    scene.spaces.insert(
        overlay_id,
        RenderSpaceState {
            id: overlay_id,
            is_active: true,
            is_overlay: true,
            override_view_position: true,
            root_transform: crate::shared::RenderTransform {
                position: Vec3::new(2.0, 0.0, 0.0),
                scale: Vec3::ONE,
                rotation: Quat::IDENTITY,
            },
            view_transform: crate::shared::RenderTransform {
                position: Vec3::new(99.0, 0.0, 0.0),
                scale: Vec3::ONE,
                rotation: Quat::IDENTITY,
            },
            ..Default::default()
        },
    );

    let overlay = scene.space(overlay_id).expect("overlay space");
    let main = scene.active_main_space().expect("main space");
    let v_overlay_rule = view_matrix_for_world_mesh_render_space(&scene, overlay);
    let v_main = view_matrix_from_render_transform(&main.view_transform);
    let diff = (v_overlay_rule - v_main).to_cols_array();
    let err: f32 = diff.iter().map(|&x| x.abs()).sum();
    assert!(
        err < 1e-4,
        "overlay space view matrix must match main space (got err sum {err})"
    );

    let v_from_overlay_only = view_matrix_from_render_transform(&overlay.view_transform);
    let diff_wrong = (v_overlay_rule - v_from_overlay_only).to_cols_array();
    let err_wrong: f32 = diff_wrong.iter().map(|&x| x.abs()).sum();
    assert!(
        err_wrong > 0.1,
        "sanity: overlay-only view must differ from main when positions differ"
    );
}
