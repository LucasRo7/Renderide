//! Resolves the active skybox cubemap used as frame-global indirect specular.

use crate::assets::asset_transfer_queue::AssetTransferQueue;
use crate::assets::material::{
    MaterialPropertyLookupIds, MaterialPropertyStore, MaterialPropertyValue, PropertyIdRegistry,
};
use crate::assets::texture::{unpack_host_texture_packed, HostTextureAssetKind};
use crate::backend::frame_gpu::SkyboxSpecularEnvironmentSource;
use crate::backend::MaterialSystem;
use crate::materials::RasterPipelineKind;
use crate::scene::SceneCoordinator;

/// Resolves the active main render space's cubemap-backed skybox into a frame-global specular source.
pub(crate) fn resolve_active_main_skybox_specular_environment(
    scene: &SceneCoordinator,
    materials: &MaterialSystem,
    assets: &AssetTransferQueue,
) -> Option<SkyboxSpecularEnvironmentSource> {
    let material_asset_id = active_main_skybox_material_asset_id(scene)?;
    let store = materials.material_property_store();
    let shader_asset_id = store.shader_asset_for_material(material_asset_id)?;
    let route_name = shader_route_name(materials, shader_asset_id)?;
    if !skybox_route_supports_specular_cubemap(&route_name) {
        return None;
    }

    let lookup = MaterialPropertyLookupIds {
        material_asset_id,
        mesh_property_block_slot0: None,
    };
    let asset_id =
        resolve_projection360_cubemap_asset_id(store, materials.property_id_registry(), lookup)?;
    let cubemap = assets.cubemap_pool.get_texture(asset_id)?;
    if cubemap.mip_levels_resident == 0 {
        return None;
    }

    Some(SkyboxSpecularEnvironmentSource {
        asset_id,
        view: cubemap.view.clone(),
        sampler: cubemap.sampler.clone(),
        mip_levels_resident: cubemap.mip_levels_resident,
        storage_v_inverted: cubemap.storage_v_inverted,
    })
}

/// Returns the skybox material id from the active non-overlay render space.
fn active_main_skybox_material_asset_id(scene: &SceneCoordinator) -> Option<i32> {
    let material_asset_id = scene.active_main_space()?.skybox_material_asset_id;
    (material_asset_id >= 0).then_some(material_asset_id)
}

/// Returns a shader route name or stem for a shader asset id.
fn shader_route_name(materials: &MaterialSystem, shader_asset_id: i32) -> Option<String> {
    let registry = materials.material_registry()?;
    if let Some(stem) = registry.stem_for_shader_asset(shader_asset_id) {
        return Some(stem.to_string());
    }
    registry
        .shader_routes_for_hud()
        .into_iter()
        .find(|(id, _, _)| *id == shader_asset_id)
        .and_then(|(_, pipeline, display)| match pipeline {
            RasterPipelineKind::EmbeddedStem(stem) => Some(stem.to_string()),
            RasterPipelineKind::Null => display,
        })
}

/// True for the cubemap-capable skybox route handled by this v1 environment binding.
fn skybox_route_supports_specular_cubemap(route_name: &str) -> bool {
    route_name.to_ascii_lowercase().contains("projection360")
}

/// Resolves the primary cubemap asset id from a Projection360 skybox material.
fn resolve_projection360_cubemap_asset_id(
    store: &MaterialPropertyStore,
    registry: &PropertyIdRegistry,
    lookup: MaterialPropertyLookupIds,
) -> Option<i32> {
    let main_cube = property_texture(store, registry, lookup, "_MainCube")
        .or_else(|| property_texture(store, registry, lookup, "_Cube"));
    if let Some((asset_id, HostTextureAssetKind::Cubemap)) = main_cube {
        return Some(asset_id);
    }

    match property_texture(store, registry, lookup, "_MainTex")
        .or_else(|| property_texture(store, registry, lookup, "_Tex"))
    {
        Some((asset_id, HostTextureAssetKind::Cubemap)) => Some(asset_id),
        _ => None,
    }
}

/// Reads a packed texture property by host name.
fn property_texture(
    store: &MaterialPropertyStore,
    registry: &PropertyIdRegistry,
    lookup: MaterialPropertyLookupIds,
    name: &str,
) -> Option<(i32, HostTextureAssetKind)> {
    let pid = registry.intern(name);
    match store.get_merged(lookup, pid) {
        Some(MaterialPropertyValue::Texture(packed)) => unpack_host_texture_packed(*packed),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::ipc::SharedMemoryAccessor;
    use crate::scene::SceneCoordinator;
    use crate::shared::{FrameSubmitData, RenderSpaceUpdate};

    /// Packs a host texture id using the same high-bit asset-kind tag as the shared host packer.
    fn pack_host_texture(asset_id: i32, kind: HostTextureAssetKind) -> i32 {
        let type_bits = 3u32;
        let pack_type_shift = 32u32 - type_bits;
        ((asset_id as u32) | ((kind as u32) << pack_type_shift)) as i32
    }

    /// Creates a lookup and empty material property store for resolver tests.
    fn store_and_lookup(
        material_asset_id: i32,
    ) -> (MaterialPropertyStore, MaterialPropertyLookupIds) {
        (
            MaterialPropertyStore::new(),
            MaterialPropertyLookupIds {
                material_asset_id,
                mesh_property_block_slot0: None,
            },
        )
    }

    #[test]
    fn projection360_prefers_main_cube_over_main_tex() {
        let registry = PropertyIdRegistry::new();
        let (mut store, lookup) = store_and_lookup(7);
        store.set_material(
            lookup.material_asset_id,
            registry.intern("_MainTex"),
            MaterialPropertyValue::Texture(pack_host_texture(11, HostTextureAssetKind::Cubemap)),
        );
        store.set_material(
            lookup.material_asset_id,
            registry.intern("_MainCube"),
            MaterialPropertyValue::Texture(pack_host_texture(42, HostTextureAssetKind::Cubemap)),
        );

        assert_eq!(
            resolve_projection360_cubemap_asset_id(&store, &registry, lookup),
            Some(42)
        );
    }

    #[test]
    fn projection360_accepts_cubemap_main_tex_fallback() {
        let registry = PropertyIdRegistry::new();
        let (mut store, lookup) = store_and_lookup(8);
        store.set_material(
            lookup.material_asset_id,
            registry.intern("_MainTex"),
            MaterialPropertyValue::Texture(pack_host_texture(13, HostTextureAssetKind::Cubemap)),
        );

        assert_eq!(
            resolve_projection360_cubemap_asset_id(&store, &registry, lookup),
            Some(13)
        );
    }

    #[test]
    fn projection360_rejects_non_cubemap_textures() {
        let registry = PropertyIdRegistry::new();
        let (mut store, lookup) = store_and_lookup(9);
        store.set_material(
            lookup.material_asset_id,
            registry.intern("_MainCube"),
            MaterialPropertyValue::Texture(pack_host_texture(15, HostTextureAssetKind::Texture2D)),
        );

        assert_eq!(
            resolve_projection360_cubemap_asset_id(&store, &registry, lookup),
            None
        );
    }

    #[test]
    fn only_projection360_routes_support_v1_specular_cubemaps() {
        assert!(skybox_route_supports_specular_cubemap(
            "skybox_projection360_default"
        ));
        assert!(!skybox_route_supports_specular_cubemap(
            "skybox_gradientskybox_default"
        ));
    }

    #[test]
    fn active_main_skybox_uses_lowest_active_non_overlay_space() {
        let mut scene = SceneCoordinator::new();
        let mut shm = SharedMemoryAccessor::new(String::new());
        scene
            .apply_frame_submit(
                &mut shm,
                &FrameSubmitData {
                    render_spaces: vec![
                        RenderSpaceUpdate {
                            id: 10,
                            is_active: true,
                            is_overlay: false,
                            skybox_material_asset_id: 100,
                            ..RenderSpaceUpdate::default()
                        },
                        RenderSpaceUpdate {
                            id: 2,
                            is_active: true,
                            is_overlay: false,
                            skybox_material_asset_id: 200,
                            ..RenderSpaceUpdate::default()
                        },
                        RenderSpaceUpdate {
                            id: 1,
                            is_active: true,
                            is_overlay: true,
                            skybox_material_asset_id: 300,
                            ..RenderSpaceUpdate::default()
                        },
                    ],
                    ..FrameSubmitData::default()
                },
            )
            .expect("empty render-space headers should apply without shared buffers");

        assert_eq!(active_main_skybox_material_asset_id(&scene), Some(200));
    }
}
