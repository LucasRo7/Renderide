use super::GraphBuilder;
use crate::render_graph::error::{GraphBuildError, RenderPassError, SetupError};
use crate::render_graph::ids::PassId;
use crate::render_graph::pass::{GroupScope, PassBuilder, PassPhase, RenderPass};
use crate::render_graph::resources::{
    BufferAccess, BufferHandle, BufferImportSource, BufferSizePolicy, FrameTargetRole,
    HistorySlotId, ImportedBufferDecl, ImportedBufferHandle, ImportedTextureDecl,
    ImportedTextureHandle, StorageAccess, TextureAccess, TextureAttachmentResolve,
    TextureAttachmentTarget, TextureHandle, TextureResourceHandle, TransientBufferDesc,
    TransientExtent, TransientTextureDesc, TransientTextureFormat,
};
use crate::render_graph::{PassKind, RenderPassContext};

struct TestPass {
    name: &'static str,
    phase: PassPhase,
    kind: PassKind,
    texture_reads: Vec<TextureHandle>,
    texture_writes: Vec<TextureHandle>,
    buffer_reads: Vec<BufferHandle>,
    buffer_writes: Vec<BufferHandle>,
    imported_texture_writes: Vec<ImportedTextureHandle>,
    imported_buffer_writes: Vec<ImportedBufferHandle>,
    raster_color: Option<TextureResourceHandle>,
    cull_exempt: bool,
}

impl TestPass {
    fn compute(name: &'static str) -> Self {
        Self {
            name,
            phase: PassPhase::PerView,
            kind: PassKind::Compute,
            texture_reads: Vec::new(),
            texture_writes: Vec::new(),
            buffer_reads: Vec::new(),
            buffer_writes: Vec::new(),
            imported_texture_writes: Vec::new(),
            imported_buffer_writes: Vec::new(),
            raster_color: None,
            cull_exempt: false,
        }
    }

    fn raster(name: &'static str, color: impl Into<TextureResourceHandle>) -> Self {
        let mut pass = Self::compute(name);
        pass.kind = PassKind::Raster;
        pass.raster_color = Some(color.into());
        pass
    }

    fn frame_global(mut self) -> Self {
        self.phase = PassPhase::FrameGlobal;
        self
    }

    fn cull_exempt(mut self) -> Self {
        self.cull_exempt = true;
        self
    }
}

impl RenderPass for TestPass {
    fn name(&self) -> &str {
        self.name
    }

    fn phase(&self) -> PassPhase {
        self.phase
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        match self.kind {
            PassKind::Raster => {
                let mut r = b.raster();
                if let Some(color) = self.raster_color {
                    r.color(
                        color,
                        wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                        Option::<TextureResourceHandle>::None,
                    );
                }
            }
            PassKind::Compute => b.compute(),
            PassKind::Copy => b.copy(),
            PassKind::Callback => {}
        }
        if self.cull_exempt {
            b.cull_exempt();
        }
        for &h in &self.texture_reads {
            b.read_texture(
                h,
                TextureAccess::Sampled {
                    stages: wgpu::ShaderStages::COMPUTE,
                },
            );
        }
        for &h in &self.texture_writes {
            b.write_texture(h, TextureAccess::CopyDst);
        }
        for &h in &self.buffer_reads {
            b.read_buffer(
                h,
                BufferAccess::Storage {
                    stages: wgpu::ShaderStages::COMPUTE,
                    access: StorageAccess::ReadOnly,
                },
            );
        }
        for &h in &self.buffer_writes {
            b.write_buffer(h, BufferAccess::CopyDst);
        }
        for &h in &self.imported_texture_writes {
            b.import_texture(h, TextureAccess::Present);
        }
        for &h in &self.imported_buffer_writes {
            b.import_buffer(h, BufferAccess::CopyDst);
        }
        Ok(())
    }

    fn execute(&mut self, _ctx: &mut RenderPassContext<'_, '_, '_>) -> Result<(), RenderPassError> {
        Ok(())
    }
}

fn tex_desc(label: &'static str) -> TransientTextureDesc {
    TransientTextureDesc::texture_2d(
        label,
        wgpu::TextureFormat::Rgba8Unorm,
        TransientExtent::Custom {
            width: 64,
            height: 64,
        },
        1,
        wgpu::TextureUsages::empty(),
    )
}

fn frame_sampled_tex_desc(label: &'static str) -> TransientTextureDesc {
    TransientTextureDesc::frame_sampled_texture_2d(
        label,
        wgpu::TextureFormat::Rgba8Unorm,
        TransientExtent::Custom {
            width: 64,
            height: 64,
        },
        wgpu::TextureUsages::empty(),
    )
}

fn backbuffer_import() -> ImportedTextureDecl {
    ImportedTextureDecl {
        label: "backbuffer",
        source: crate::render_graph::resources::ImportSource::FrameTarget(
            FrameTargetRole::ColorAttachment,
        ),
        initial_access: TextureAccess::ColorAttachment {
            load: wgpu::LoadOp::Load,
            store: wgpu::StoreOp::Store,
            resolve_to: None,
        },
        final_access: TextureAccess::Present,
    }
}

fn depth_import() -> ImportedTextureDecl {
    ImportedTextureDecl {
        label: "depth",
        source: crate::render_graph::resources::ImportSource::FrameTarget(
            FrameTargetRole::DepthAttachment,
        ),
        initial_access: TextureAccess::DepthAttachment {
            depth: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
            stencil: None,
        },
        final_access: TextureAccess::Sampled {
            stages: wgpu::ShaderStages::COMPUTE,
        },
    }
}

fn buffer_import_readback() -> ImportedBufferDecl {
    ImportedBufferDecl {
        label: "readback",
        source: BufferImportSource::External,
        initial_access: BufferAccess::CopyDst,
        final_access: BufferAccess::CopyDst,
    }
}

#[test]
fn linear_chain_schedules_in_order() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::new();
    let tex = b.create_texture(tex_desc("color"));
    let bb = b.import_texture(backbuffer_import());
    let mut a = TestPass::compute("a");
    a.texture_writes.push(tex);
    let mut c = TestPass::raster("c", bb);
    c.texture_reads.push(tex);
    b.add_pass(Box::new(a));
    b.add_pass(Box::new(c));
    let g = b.build()?;
    assert_eq!(g.pass_count(), 2);
    assert_eq!(g.pass_info[0].name, "a");
    assert_eq!(g.pass_info[1].name, "c");
    Ok(())
}

#[test]
fn parallel_passes_single_level() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::new();
    let out_a = b.import_texture(backbuffer_import());
    let out_b = b.import_buffer(buffer_import_readback());
    b.add_pass(Box::new(TestPass::raster("a", out_a)));
    let mut b_pass = TestPass::compute("b");
    b_pass.imported_buffer_writes.push(out_b);
    b.add_pass(Box::new(b_pass));
    let g = b.build()?;
    assert_eq!(g.compile_stats.topo_levels, 1);
    assert_eq!(g.pass_count(), 2);
    Ok(())
}

#[test]
fn cycle_detected_through_handle_rw_conflict() {
    let mut b = GraphBuilder::new();
    let tex = b.create_texture(tex_desc("color"));
    let bb = b.import_texture(backbuffer_import());
    let mut a = TestPass::raster("a", bb);
    a.texture_reads.push(tex);
    let mut c = TestPass::compute("c");
    c.texture_writes.push(tex);
    let a_id = b.add_pass(Box::new(a));
    let c_id = b.add_pass(Box::new(c));
    b.add_edge(a_id, c_id);
    assert!(matches!(
        b.build(),
        Err(GraphBuildError::MissingDependency { .. })
    ));
}

#[test]
fn read_without_writer_errors_with_handle_and_access() {
    let mut b = GraphBuilder::new();
    let tex = b.create_texture(tex_desc("orphan"));
    let mut p = TestPass::compute("reader");
    p.texture_reads.push(tex);
    b.add_pass(Box::new(p));
    assert!(matches!(
        b.build(),
        Err(GraphBuildError::MissingDependency { .. })
    ));
}

#[test]
fn aliased_handles_share_slot_when_lifetimes_disjoint() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::new();
    let a = b.create_texture(tex_desc("a"));
    let c = b.create_texture(tex_desc("c"));
    let bb = b.import_texture(backbuffer_import());
    let mut p0 = TestPass::compute("write-a");
    p0.texture_writes.push(a);
    let mut p1 = TestPass::raster("export-a", bb);
    p1.texture_reads.push(a);
    let mut p2 = TestPass::compute("write-c");
    p2.texture_writes.push(c);
    let mut p3 = TestPass::raster("export-c", bb);
    p3.texture_reads.push(c);
    b.add_pass(Box::new(p0));
    let p1_id = b.add_pass(Box::new(p1));
    let p2_id = b.add_pass(Box::new(p2));
    b.add_pass(Box::new(p3));
    b.add_edge(p1_id, p2_id);
    let g = b.build()?;
    assert_eq!(
        g.transient_textures[a.index()].physical_slot,
        g.transient_textures[c.index()].physical_slot
    );
    assert_eq!(g.compile_stats.transient_texture_slots, 1);
    Ok(())
}

#[test]
fn aliased_handles_do_not_share_when_desc_alias_false() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::new();
    let mut d0 = tex_desc("a");
    let mut d1 = tex_desc("c");
    d0.alias = false;
    d1.alias = false;
    let a = b.create_texture(d0);
    let c = b.create_texture(d1);
    let bb = b.import_texture(backbuffer_import());
    let mut p0 = TestPass::compute("write-a");
    p0.texture_writes.push(a);
    let mut p1 = TestPass::raster("export-a", bb);
    p1.texture_reads.push(a);
    let mut p2 = TestPass::compute("write-c");
    p2.texture_writes.push(c);
    let mut p3 = TestPass::raster("export-c", bb);
    p3.texture_reads.push(c);
    b.add_pass(Box::new(p0));
    let p1_id = b.add_pass(Box::new(p1));
    let p2_id = b.add_pass(Box::new(p2));
    b.add_pass(Box::new(p3));
    b.add_edge(p1_id, p2_id);
    let g = b.build()?;
    assert_ne!(
        g.transient_textures[a.index()].physical_slot,
        g.transient_textures[c.index()].physical_slot
    );
    Ok(())
}

#[test]
fn usage_union_promotes_transient_to_storage_when_sampled_and_stored() -> Result<(), GraphBuildError>
{
    let mut b = GraphBuilder::new();
    let tex = b.create_texture(tex_desc("scratch"));
    let bb = b.import_texture(backbuffer_import());
    let mut p0 = TestPass::compute("write");
    p0.texture_writes.push(tex);
    let mut p1 = TestPass::raster("export", bb);
    p1.texture_reads.push(tex);
    b.add_pass(Box::new(p0));
    b.add_pass(Box::new(p1));
    let g = b.build()?;
    let usage = g.transient_textures[tex.index()].usage;
    assert!(usage.contains(wgpu::TextureUsages::COPY_DST));
    assert!(usage.contains(wgpu::TextureUsages::TEXTURE_BINDING));
    Ok(())
}

#[test]
fn dead_pass_culled_when_output_unused() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::new();
    let tex = b.create_texture(tex_desc("dead"));
    let mut p = TestPass::compute("dead");
    p.texture_writes.push(tex);
    b.add_pass(Box::new(p));
    let g = b.build()?;
    assert_eq!(g.pass_count(), 0);
    assert_eq!(g.compile_stats.culled_count, 1);
    Ok(())
}

#[test]
fn dead_pass_retained_when_marked_exempt() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::new();
    b.add_pass(Box::new(TestPass::compute("side-effect").cull_exempt()));
    let g = b.build()?;
    assert_eq!(g.pass_count(), 1);
    Ok(())
}

#[test]
fn raster_pass_without_attachments_rejected() {
    let mut b = GraphBuilder::new();
    let mut p = TestPass::compute("bad");
    p.kind = PassKind::Raster;
    b.add_pass(Box::new(p));
    assert!(matches!(
        b.build(),
        Err(GraphBuildError::Setup {
            source: SetupError::RasterWithoutAttachments,
            ..
        })
    ));
}

#[test]
fn compute_pass_with_attachment_rejected() {
    struct BadPass(ImportedTextureHandle);
    impl RenderPass for BadPass {
        fn name(&self) -> &str {
            "bad"
        }
        fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
            b.compute();
            b.import_texture(
                self.0,
                TextureAccess::ColorAttachment {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                    resolve_to: None,
                },
            );
            Ok(())
        }
        fn execute(
            &mut self,
            _ctx: &mut RenderPassContext<'_, '_, '_>,
        ) -> Result<(), RenderPassError> {
            Ok(())
        }
    }
    let mut b = GraphBuilder::new();
    let bb = b.import_texture(backbuffer_import());
    b.add_pass(Box::new(BadPass(bb)));
    assert!(matches!(
        b.build(),
        Err(GraphBuildError::Setup {
            source: SetupError::NonRasterPassHasAttachment,
            ..
        })
    ));
}

#[test]
fn frameglobal_runs_before_perview_by_default() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::new();
    let bb = b.import_texture(backbuffer_import());
    b.add_pass(Box::new(TestPass::raster("per-view", bb)));
    b.add_pass(Box::new(
        TestPass::compute("frame").frame_global().cull_exempt(),
    ));
    let g = b.build()?;
    assert_eq!(g.pass_info[0].name, "frame");
    assert_eq!(g.pass_info[1].name, "per-view");
    Ok(())
}

#[test]
fn group_order_respects_group_after_declarations() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::new();
    let bb = b.import_texture(backbuffer_import());
    let a_group = b.group("a", GroupScope::PerView);
    let z_group = b.group("z", GroupScope::PerView);
    b.group_after(z_group, a_group);
    b.add_pass_to_group(z_group, Box::new(TestPass::raster("z", bb)));
    b.add_pass_to_group(a_group, Box::new(TestPass::compute("a").cull_exempt()));
    let g = b.build()?;
    assert_eq!(g.pass_info[0].name, "a");
    assert_eq!(g.pass_info[1].name, "z");
    Ok(())
}

#[test]
fn multiview_mask_propagates_into_template() -> Result<(), GraphBuildError> {
    struct MvPass(ImportedTextureHandle);
    impl RenderPass for MvPass {
        fn name(&self) -> &str {
            "mv"
        }
        fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
            let mut r = b.raster();
            r.color(
                self.0,
                wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                Option::<TextureResourceHandle>::None,
            );
            r.multiview(
                std::num::NonZeroU32::new(3)
                    .ok_or_else(|| SetupError::Message("multiview test mask".to_string()))?,
            );
            Ok(())
        }
        fn execute(
            &mut self,
            _ctx: &mut RenderPassContext<'_, '_, '_>,
        ) -> Result<(), RenderPassError> {
            Ok(())
        }
    }
    let mut b = GraphBuilder::new();
    let bb = b.import_texture(backbuffer_import());
    b.add_pass(Box::new(MvPass(bb)));
    let g = b.build()?;
    assert_eq!(g.pass_info[0].multiview_mask.unwrap().get(), 3);
    let mv = g.pass_info[0]
        .raster_template
        .as_ref()
        .and_then(|template| template.multiview_mask);
    assert!(mv.is_some());
    assert_eq!(mv.unwrap().get(), 3);
    Ok(())
}

#[test]
fn raster_template_records_color_depth_and_resolve_targets() -> Result<(), GraphBuildError> {
    struct RasterTemplatePass {
        color: ImportedTextureHandle,
        resolve: ImportedTextureHandle,
        depth: ImportedTextureHandle,
    }
    impl RenderPass for RasterTemplatePass {
        fn name(&self) -> &str {
            "templated"
        }
        fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
            let mut r = b.raster();
            r.color(
                self.color,
                wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                    store: wgpu::StoreOp::Discard,
                },
                Some(self.resolve),
            );
            r.depth(
                self.depth,
                wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.5),
                    store: wgpu::StoreOp::Store,
                },
                None,
            );
            Ok(())
        }
        fn execute(
            &mut self,
            _ctx: &mut RenderPassContext<'_, '_, '_>,
        ) -> Result<(), RenderPassError> {
            Ok(())
        }
    }

    let mut b = GraphBuilder::new();
    let color = b.import_texture(backbuffer_import());
    let resolve = b.import_texture(backbuffer_import());
    let depth = b.import_texture(depth_import());
    b.add_pass(Box::new(RasterTemplatePass {
        color,
        resolve,
        depth,
    }));

    let g = b.build()?;
    assert!(g.pass_info[0].raster_template.is_some());
    let template = g.pass_info[0].raster_template.as_ref().unwrap();
    assert_eq!(template.color_attachments.len(), 1);
    assert_eq!(
        template.color_attachments[0].target,
        TextureAttachmentTarget::Resource(TextureResourceHandle::Imported(color))
    );
    assert_eq!(
        template.color_attachments[0].resolve_to,
        Some(TextureAttachmentResolve::Always(
            TextureResourceHandle::Imported(resolve)
        ))
    );
    assert_eq!(template.color_attachments[0].store, wgpu::StoreOp::Discard);
    assert_eq!(
        template.depth_stencil_attachment.as_ref().map(|d| d.target),
        Some(TextureAttachmentTarget::Resource(
            TextureResourceHandle::Imported(depth)
        ))
    );
    Ok(())
}

#[test]
fn raster_template_records_frame_sampled_targets() -> Result<(), GraphBuildError> {
    struct FrameSampledRasterPass {
        color: ImportedTextureHandle,
        resolve: ImportedTextureHandle,
        depth: ImportedTextureHandle,
        msaa_color: TextureHandle,
        msaa_depth: TextureHandle,
    }
    impl RenderPass for FrameSampledRasterPass {
        fn name(&self) -> &str {
            "frame-sampled"
        }
        fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
            let mut r = b.raster();
            r.frame_sampled_color(
                self.color,
                self.msaa_color,
                wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                    store: wgpu::StoreOp::Store,
                },
                Some(self.resolve),
            );
            r.frame_sampled_depth(
                self.depth,
                self.msaa_depth,
                wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.5),
                    store: wgpu::StoreOp::Store,
                },
                None,
            );
            Ok(())
        }
        fn execute(
            &mut self,
            _ctx: &mut RenderPassContext<'_, '_, '_>,
        ) -> Result<(), RenderPassError> {
            Ok(())
        }
    }

    let mut b = GraphBuilder::new();
    let color = b.import_texture(backbuffer_import());
    let resolve = b.import_texture(backbuffer_import());
    let depth = b.import_texture(depth_import());
    let msaa_color = b.create_texture(frame_sampled_tex_desc("msaa-color"));
    let msaa_depth = b.create_texture(TransientTextureDesc::frame_sampled_texture_2d(
        "msaa-depth",
        wgpu::TextureFormat::Depth32Float,
        TransientExtent::Custom {
            width: 64,
            height: 64,
        },
        wgpu::TextureUsages::empty(),
    ));
    b.add_pass(Box::new(FrameSampledRasterPass {
        color,
        resolve,
        depth,
        msaa_color,
        msaa_depth,
    }));

    let g = b.build()?;
    assert!(g.pass_info[0].raster_template.is_some());
    let template = g.pass_info[0].raster_template.as_ref().unwrap();
    assert_eq!(
        template.color_attachments[0].target,
        TextureAttachmentTarget::FrameSampled {
            single_sample: TextureResourceHandle::Imported(color),
            multisampled: TextureResourceHandle::Transient(msaa_color),
        }
    );
    assert_eq!(
        template.color_attachments[0].resolve_to,
        Some(TextureAttachmentResolve::FrameMultisampled(
            TextureResourceHandle::Imported(resolve)
        ))
    );
    assert_eq!(
        template.depth_stencil_attachment.as_ref().map(|d| d.target),
        Some(TextureAttachmentTarget::FrameSampled {
            single_sample: TextureResourceHandle::Imported(depth),
            multisampled: TextureResourceHandle::Transient(msaa_depth),
        })
    );
    Ok(())
}

#[test]
fn buffer_aliasing_uses_size_and_usage_key() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::new();
    let a = b.create_buffer(TransientBufferDesc {
        label: "a",
        size_policy: BufferSizePolicy::Fixed(64),
        base_usage: wgpu::BufferUsages::empty(),
        alias: true,
    });
    let c = b.create_buffer(TransientBufferDesc {
        label: "c",
        size_policy: BufferSizePolicy::Fixed(64),
        base_usage: wgpu::BufferUsages::empty(),
        alias: true,
    });
    let out = b.import_buffer(ImportedBufferDecl {
        label: "history",
        source: BufferImportSource::PingPong(HistorySlotId::HiZ),
        initial_access: BufferAccess::CopyDst,
        final_access: BufferAccess::CopyDst,
    });
    let mut p0 = TestPass::compute("write-a");
    p0.buffer_writes.push(a);
    let mut p1 = TestPass::compute("export-a");
    p1.buffer_reads.push(a);
    p1.imported_buffer_writes.push(out);
    let mut p2 = TestPass::compute("write-c");
    p2.buffer_writes.push(c);
    let mut p3 = TestPass::compute("export-c");
    p3.buffer_reads.push(c);
    p3.imported_buffer_writes.push(out);
    b.add_pass(Box::new(p0));
    let p1_id = b.add_pass(Box::new(p1));
    let p2_id = b.add_pass(Box::new(p2));
    b.add_pass(Box::new(p3));
    b.add_edge(p1_id, p2_id);
    let g = b.build()?;
    assert_eq!(
        g.transient_buffers[a.index()].physical_slot,
        g.transient_buffers[c.index()].physical_slot
    );
    Ok(())
}

#[test]
fn texture_aliasing_keys_on_sample_count_policy() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::new();
    let frame_sampled = b.create_texture(frame_sampled_tex_desc("frame-sampled"));
    let fixed = b.create_texture(tex_desc("fixed"));
    let bb = b.import_texture(backbuffer_import());
    let mut p0 = TestPass::compute("write-frame-sampled");
    p0.texture_writes.push(frame_sampled);
    let mut p1 = TestPass::raster("export-frame-sampled", bb);
    p1.texture_reads.push(frame_sampled);
    let mut p2 = TestPass::compute("write-fixed");
    p2.texture_writes.push(fixed);
    let mut p3 = TestPass::raster("export-fixed", bb);
    p3.texture_reads.push(fixed);
    b.add_pass(Box::new(p0));
    let p1_id = b.add_pass(Box::new(p1));
    b.add_pass(Box::new(p2));
    b.add_pass(Box::new(p3));
    b.add_edge(p1_id, PassId(2));

    let g = b.build()?;
    assert_ne!(
        g.transient_textures[frame_sampled.index()].physical_slot,
        g.transient_textures[fixed.index()].physical_slot
    );
    Ok(())
}

#[test]
fn frame_sample_count_policy_resolves_current_frame_value() {
    use crate::render_graph::resources::TransientSampleCount;
    assert_eq!(TransientSampleCount::Fixed(0).resolve(4), 1);
    assert_eq!(TransientSampleCount::Fixed(2).resolve(4), 2);
    assert_eq!(TransientSampleCount::Frame.resolve(0), 1);
    assert_eq!(TransientSampleCount::Frame.resolve(4), 4);
}

#[test]
fn frame_texture_format_and_layer_policies_resolve_current_frame_values() {
    assert_eq!(
        TransientTextureFormat::Fixed(wgpu::TextureFormat::Rgba8Unorm).resolve(
            wgpu::TextureFormat::Bgra8UnormSrgb,
            wgpu::TextureFormat::Depth24PlusStencil8,
            wgpu::TextureFormat::Rgba16Float,
        ),
        wgpu::TextureFormat::Rgba8Unorm
    );
    assert_eq!(
        TransientTextureFormat::FrameColor.resolve(
            wgpu::TextureFormat::Bgra8UnormSrgb,
            wgpu::TextureFormat::Depth24PlusStencil8,
            wgpu::TextureFormat::Rgba16Float,
        ),
        wgpu::TextureFormat::Bgra8UnormSrgb
    );
    assert_eq!(
        TransientTextureFormat::FrameDepthStencil.resolve(
            wgpu::TextureFormat::Bgra8UnormSrgb,
            wgpu::TextureFormat::Depth24PlusStencil8,
            wgpu::TextureFormat::Rgba16Float,
        ),
        wgpu::TextureFormat::Depth24PlusStencil8
    );
    assert_eq!(
        TransientTextureFormat::SceneColorHdr.resolve(
            wgpu::TextureFormat::Bgra8UnormSrgb,
            wgpu::TextureFormat::Depth24PlusStencil8,
            wgpu::TextureFormat::Rg11b10Ufloat,
        ),
        wgpu::TextureFormat::Rg11b10Ufloat
    );
    use crate::render_graph::resources::TransientArrayLayers;
    assert_eq!(TransientArrayLayers::Fixed(0).resolve(true), 1);
    assert_eq!(TransientArrayLayers::Fixed(3).resolve(false), 3);
    assert_eq!(TransientArrayLayers::Frame.resolve(false), 1);
    assert_eq!(TransientArrayLayers::Frame.resolve(true), 2);
}
