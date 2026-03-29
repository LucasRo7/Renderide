//! Canonical essential WGSL shader programs resolved from uploaded Unity shader names.

/// Small supported shader set for the renderer's native WGSL implementations.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum EssentialShaderProgram {
    /// No native WGSL equivalent is registered for this shader name.
    Unsupported,
    /// Resonite `PBSMetallic` family.
    PbsMetallic,
    /// Resonite world `Shader "Unlit"`.
    WorldUnlit,
    /// Resonite `UI/Unlit`.
    UiUnlit,
    /// Resonite `UI/Text/Unlit`.
    UiTextUnlit,
}

fn compact_alnum_lower(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

fn is_pbs_family(key: &str) -> bool {
    key.starts_with("pbs") || key.starts_with("paintpbs")
}

fn is_toon_lit_family(key: &str) -> bool {
    key.starts_with("xstoon") || key.starts_with("toon")
}

fn is_ui_unlit_family(key: &str) -> bool {
    key == "uiunlit" || key == "uicirclesegment"
}

fn is_ui_text_family(key: &str) -> bool {
    key == "uitextunlit" || key == "textunlit"
}

fn is_world_unlit_family(key: &str) -> bool {
    key == "unlit"
        || key.ends_with("unlit")
        || matches!(
            key,
            "overlayunlit"
                | "fresnel"
                | "fresnellerp"
                | "overlayfresnel"
                | "matcap"
                | "projection360"
                | "cubemapprojection"
                | "reflection"
                | "proceduralskybox"
                | "uvrect"
                | "billboardunlit"
                | "volumeunlit"
                | "wireframeunlittransition"
        )
}

/// Resolves the essential WGSL program from a Unity shader name or plain upload label.
pub fn resolve_essential_shader_program(name: Option<&str>) -> EssentialShaderProgram {
    let Some(name) = name else {
        return EssentialShaderProgram::Unsupported;
    };
    let Some(token) = name.split_whitespace().next() else {
        return EssentialShaderProgram::Unsupported;
    };
    let key = compact_alnum_lower(token);
    if is_ui_text_family(&key) {
        EssentialShaderProgram::UiTextUnlit
    } else if is_ui_unlit_family(&key) {
        EssentialShaderProgram::UiUnlit
    } else if is_pbs_family(&key) || is_toon_lit_family(&key) {
        EssentialShaderProgram::PbsMetallic
    } else if is_world_unlit_family(&key) {
        EssentialShaderProgram::WorldUnlit
    } else {
        EssentialShaderProgram::Unsupported
    }
}

#[cfg(test)]
mod tests {
    use super::{EssentialShaderProgram, resolve_essential_shader_program};

    #[test]
    fn resolves_essential_programs() {
        assert_eq!(
            resolve_essential_shader_program(Some("PBSMetallic")),
            EssentialShaderProgram::PbsMetallic
        );
        assert_eq!(
            resolve_essential_shader_program(Some("Unlit")),
            EssentialShaderProgram::WorldUnlit
        );
        assert_eq!(
            resolve_essential_shader_program(Some("UI_Unlit")),
            EssentialShaderProgram::UiUnlit
        );
        assert_eq!(
            resolve_essential_shader_program(Some("UI_TextUnlit")),
            EssentialShaderProgram::UiTextUnlit
        );
        assert_eq!(
            resolve_essential_shader_program(Some("PBSLerpSpecular")),
            EssentialShaderProgram::PbsMetallic
        );
        assert_eq!(
            resolve_essential_shader_program(Some("OverlayUnlit")),
            EssentialShaderProgram::WorldUnlit
        );
        assert_eq!(
            resolve_essential_shader_program(Some("TextUnlit")),
            EssentialShaderProgram::UiTextUnlit
        );
        assert_eq!(
            resolve_essential_shader_program(Some("XSToon2.0")),
            EssentialShaderProgram::PbsMetallic
        );
    }
}
