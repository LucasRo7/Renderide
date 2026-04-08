//! Composes `shaders/source/modules/*.wgsl` with [`naga_oil`] (`#import`), validates with naga, and
//! writes flat `shaders/target/*.wgsl` plus `OUT_DIR/embedded_shaders.rs`.
//!
//! Optional per-material metadata: `shaders/source/materials/<stem>.meta.json` may include:
//! - `unity_names`: Unity `Shader "…"` keys (normalized at resolve time)
//! - `material_properties`, `vertex_streams`, `uniform_derived`: copied into `shaders/target/manifest.json`
//!
//! Multiview sources (`debug_world_normals`, `world_unlit`) share one config path below; `world_unlit`
//! attaches `unity_names` only to `*_default` targets (multiview targets list empty `unity_names`).
//!
//! ## Multiview variants (`*_default` / `*_multiview`)
//!
//! Sources use `#ifdef MULTIVIEW` … `#else` … `#endif`. In naga-oil, `#ifdef NAME` is true when
//! `NAME` exists in the compose [`ShaderDefValue`] map **regardless of value** — so
//! `MULTIVIEW` → `ShaderDefValue::Bool(false)` still
//! enables the `#ifdef` branch. For the non-multiview target, **omit** `MULTIVIEW` from `shader_defs`
//! entirely; for the multiview target, set `MULTIVIEW` to [`ShaderDefValue::Bool(true)`].
//!
//! Alternatively, WGSL could use `#if MULTIVIEW == true` with both `Bool(true)` and `Bool(false)`
//! in the map; the omit-key approach matches existing `#ifdef` in source materials without edits.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use naga::back::wgsl::WriterFlags;
use naga::valid::{Capabilities, ValidationFlags, Validator};
use naga_oil::compose::{
    ComposableModuleDescriptor, Composer, NagaModuleDescriptor, ShaderDefValue, ShaderLanguage,
    ShaderType,
};

/// Shader defs for material sources that use `#ifdef MULTIVIEW`.
///
/// When `enable` is false, returns an empty map so `#ifdef MULTIVIEW` is not taken. When true,
/// inserts `MULTIVIEW` as [`ShaderDefValue::Bool(true)`].
fn multiview_shader_defs(enable: bool) -> HashMap<String, ShaderDefValue> {
    let mut defs = HashMap::new();
    if enable {
        defs.insert("MULTIVIEW".to_string(), ShaderDefValue::Bool(true));
    }
    defs
}

fn validate_and_write_wgsl(
    module: &naga::Module,
    label: &str,
    out_path: &std::path::Path,
    expect_view_index: Option<bool>,
) {
    let has_vs = module
        .entry_points
        .iter()
        .any(|e| e.stage == naga::ShaderStage::Vertex && e.name == "vs_main");
    let has_fs = module
        .entry_points
        .iter()
        .any(|e| e.stage == naga::ShaderStage::Fragment && e.name == "fs_main");
    if !has_vs || !has_fs {
        panic!(
            "{label}: expected entry points vs_main and fs_main (vertex={has_vs} fragment={has_fs})",
        );
    }

    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
    let info = validator
        .validate(module)
        .unwrap_or_else(|e| panic!("validate {label}: {e}"));
    let wgsl = naga::back::wgsl::write_string(module, &info, WriterFlags::EXPLICIT_TYPES)
        .unwrap_or_else(|e| panic!("wgsl out {label}: {e}"));
    if let Some(want) = expect_view_index {
        let has = wgsl.contains("@builtin(view_index)");
        if want != has {
            panic!(
                "{label}: expected @builtin(view_index) {} in output (multiview shader_defs contract)",
                if want { "present" } else { "absent" }
            );
        }
    }
    fs::write(out_path, &wgsl).unwrap_or_else(|e| panic!("write {}: {e}", out_path.display()));
}

fn add_globals_module(composer: &mut Composer, globals_source: &str) {
    composer
        .add_composable_module(ComposableModuleDescriptor {
            source: globals_source,
            file_path: "shaders/source/modules/globals.wgsl",
            language: ShaderLanguage::Wgsl,
            ..Default::default()
        })
        .unwrap_or_else(|e| panic!("add globals module: {e}"));
}

fn compose_material(
    globals_source: &str,
    material_source: &str,
    material_file_path: &str,
    shader_defs: HashMap<String, ShaderDefValue>,
) -> naga::Module {
    let mut composer = Composer::default().with_capabilities(Capabilities::all());
    add_globals_module(&mut composer, globals_source);
    composer
        .make_naga_module(NagaModuleDescriptor {
            source: material_source,
            file_path: material_file_path,
            shader_type: ShaderType::Wgsl,
            shader_defs,
            ..Default::default()
        })
        .unwrap_or_else(|e| panic!("compose {material_file_path}: {e}"))
}

/// Reads optional JSON sidecar for a material source stem (empty object if missing).
fn read_stem_meta(materials_dir: &std::path::Path, file_stem: &str) -> serde_json::Value {
    let meta_path = materials_dir.join(format!("{file_stem}.meta.json"));
    if !meta_path.is_file() {
        return serde_json::json!({});
    }
    let raw = fs::read_to_string(&meta_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", meta_path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse {}: {e}", meta_path.display()))
}

/// Builds one `manifest.json` material entry, merging shared meta fields with per-target `stem` / `file` / `unity_names`.
fn manifest_material_entry(
    meta: &serde_json::Value,
    target_stem: &str,
    unity_names: Vec<String>,
) -> serde_json::Value {
    let mut m = serde_json::Map::new();
    m.insert("stem".to_string(), serde_json::json!(target_stem));
    m.insert(
        "file".to_string(),
        serde_json::json!(format!("shaders/target/{target_stem}.wgsl")),
    );
    m.insert("unity_names".to_string(), serde_json::json!(unity_names));
    for key in ["material_properties", "vertex_streams", "uniform_derived"] {
        if let Some(v) = meta.get(key) {
            m.insert(key.to_string(), v.clone());
        }
    }
    serde_json::Value::Object(m)
}

fn main() {
    let manifest_dir =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let globals_path = manifest_dir.join("shaders/source/modules/globals.wgsl");
    let materials_dir = manifest_dir.join("shaders/source/materials");
    let target_dir = manifest_dir.join("shaders/target");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));

    println!("cargo:rerun-if-changed=shaders/source");
    println!("cargo:rerun-if-changed=build.rs");

    fs::create_dir_all(materials_dir.parent().expect("materials parent")).expect("mkdir");
    fs::create_dir_all(&target_dir).expect("mkdir target");

    let globals_source = fs::read_to_string(&globals_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", globals_path.display()));

    let mut manifest_entries: Vec<serde_json::Value> = Vec::new();
    let mut embedded_arms = String::new();
    let mut output_stems: Vec<String> = Vec::new();

    let mut material_paths: Vec<PathBuf> = fs::read_dir(&materials_dir)
        .expect("read materials")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "wgsl").unwrap_or(false))
        .collect();
    material_paths.sort();

    for path in &material_paths {
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .expect("material stem");

        let material_source =
            fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let meta = read_stem_meta(&materials_dir, stem);
        let unity_from_meta: Vec<String> = meta
            .get("unity_names")
            .and_then(|u| u.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let multiview_pair = stem == "debug_world_normals" || stem == "world_unlit";
        if multiview_pair {
            let unity_names_default_only = stem == "world_unlit";
            let variants = [
                (format!("{stem}_default"), false),
                (format!("{stem}_multiview"), true),
            ];
            for (target_stem, multiview) in variants {
                let unity_for_entry = if unity_names_default_only && multiview {
                    Vec::new()
                } else {
                    unity_from_meta.clone()
                };
                let defs = multiview_shader_defs(multiview);
                let module = compose_material(
                    &globals_source,
                    &material_source,
                    path.to_str().expect("utf8 path"),
                    defs,
                );
                let label = format!("{target_stem} (MULTIVIEW={multiview})");
                let out_path = target_dir.join(format!("{target_stem}.wgsl"));
                validate_and_write_wgsl(&module, &label, &out_path, Some(multiview));

                manifest_entries.push(manifest_material_entry(
                    &meta,
                    &target_stem,
                    unity_for_entry,
                ));

                embedded_arms.push_str(&format!(
                    "        \"{target_stem}\" => Some(include_str!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/shaders/target/{target_stem}.wgsl\"))),\n"
                ));
                output_stems.push(target_stem);
            }
            continue;
        }

        let module = compose_material(
            &globals_source,
            &material_source,
            path.to_str().expect("utf8 path"),
            HashMap::new(),
        );
        let out_path = target_dir.join(format!("{stem}.wgsl"));
        validate_and_write_wgsl(&module, stem, &out_path, None);

        manifest_entries.push(manifest_material_entry(&meta, stem, unity_from_meta));

        embedded_arms.push_str(&format!(
            "        \"{stem}\" => Some(include_str!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/shaders/target/{stem}.wgsl\"))),\n"
        ));
        output_stems.push(stem.to_string());
    }

    let manifest = serde_json::json!({
        "materials": manifest_entries,
        "globals_module": "shaders/source/modules/globals.wgsl",
    });
    let manifest_path = target_dir.join("manifest.json");
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).expect("manifest json"),
    )
    .expect("write manifest");

    let stems_list = output_stems
        .iter()
        .map(|s| format!("    \"{s}\","))
        .collect::<Vec<_>>()
        .join("\n");

    let embedded_rs = format!(
        r#"// Generated by `build.rs` — do not edit.

/// Flattened WGSL for `stem` (see `shaders/target/{{stem}}.wgsl`).
pub fn embedded_target_wgsl(stem: &str) -> Option<&'static str> {{
    match stem {{
{embedded_arms}        _ => None,
    }}
}}

/// Stems under `shaders/target/*.wgsl` present at build time.
pub const COMPILED_MATERIAL_STEMS: &[&str] = &[
{stems}
];

/// JSON manifest written next to composed shaders.
pub const SHADER_MANIFEST_JSON: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/target/manifest.json"));
"#,
        embedded_arms = embedded_arms,
        stems = stems_list
    );

    let gen_path = out_dir.join("embedded_shaders.rs");
    fs::write(&gen_path, embedded_rs).expect("write embedded_shaders.rs");
}
