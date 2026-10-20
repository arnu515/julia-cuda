//! The model's weights as the kernels read them. Every matrix (and the embedding table) is f16 in
//! one buffer, every vector (LayerNorm weights, biases, column sums, type embeddings) f32 in
//! another; each tensor is a [`Span`] into one of the two. Prepared on the host once at load
//! ([`HostWeights::prepare`]), then uploaded with two copies ([`GpuWeights::upload`]).

use std::{marker::PhantomData, ops::Range, sync::Arc};

use color_eyre::eyre::ensure;
use cudarc::driver::{CudaSlice, CudaStream};
use half::{f16, slice::HalfFloatSliceExt};

use crate::{checkpoint::Checkpoint, prep::fold_layernorm};

/// Every span starts at a multiple of this many bytes: 16-byte vector loads and `ldmatrix` need
/// 16, and 128 is a whole cache line.
pub const ALIGN_BYTES: usize = 128;

/// Where a tensor is in the buffer of `T`s.
#[derive(Debug)]
pub struct Span<T> {
    pub start: usize,
    pub len: usize,
    _buffer: PhantomData<T>,
}

// Not derived: the derives would require `T: Clone`/`T: Copy`.
impl<T> Clone for Span<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Span<T> {}

impl<T> Span<T> {
    pub fn range(&self) -> Range<usize> {
        self.start..self.start + self.len
    }
}

/// One encoder layer's weights, matrices `[out, in]` as in the checkpoint.
#[derive(Clone, Debug)]
pub struct EncoderLayer {
    /// The attention LayerNorm's γ is folded in, except in layer 0, which has none.
    pub wqkv: Span<f16>,
    /// `colsum` for the folded `wqkv`; `None` in layer 0, whose qkv GEMM uses the plain epilogue.
    pub qkv_colsum: Option<Span<f32>>,
    pub wo: Span<f16>,
    /// The MLP LayerNorm's γ folded in.
    pub wi: Span<f16>,
    pub wi_colsum: Span<f32>,
    pub wo_mlp: Span<f16>,
    /// Global attention (else sliding window).
    pub global: bool,
}

/// One head `nn.TransformerEncoderLayer`. Its LayerNorms have biases, so they aren't folded:
/// the head is a few small GEMMs.
#[derive(Clone, Debug)]
pub struct HeadLayer {
    pub in_proj: Span<f16>,
    pub in_proj_b: Span<f32>,
    pub out_proj: Span<f16>,
    pub out_proj_b: Span<f32>,
    pub linear1: Span<f16>,
    pub linear1_b: Span<f32>,
    pub linear2: Span<f16>,
    pub linear2_b: Span<f32>,
    pub norm1_w: Span<f32>,
    pub norm1_b: Span<f32>,
    pub norm2_w: Span<f32>,
    pub norm2_b: Span<f32>,
}

#[derive(Clone, Debug)]
pub struct Scorer {
    pub norm_w: Span<f32>,
    pub norm_b: Span<f32>,
    pub lin1: Span<f16>,
    pub lin1_b: Span<f32>,
    pub lin2: Span<f16>,
    pub lin2_b: Span<f32>,
}

#[derive(Clone, Debug)]
pub struct Layout {
    pub tok_embeddings: Span<f16>,
    pub emb_norm: Span<f32>,
    pub layers: Vec<EncoderLayer>,
    pub final_norm: Span<f32>,
    /// `[N_QTYPES, D]`, added to every row of the head's input.
    pub type_emb: Span<f32>,
    pub head: Vec<HeadLayer>,
    pub scorer: Scorer,
}

/// A growing buffer that hands out aligned spans.
struct Arena<T> {
    data: Vec<T>,
}

impl<T: Copy + Default> Arena<T> {
    /// Appends `len` elements, written by `fill`.
    fn push_with(&mut self, len: usize, fill: impl FnOnce(&mut [T])) -> Span<T> {
        let start = (self.data.len()).next_multiple_of(ALIGN_BYTES / size_of::<T>());
        self.data.resize(start + len, T::default());
        fill(&mut self.data[start..]);
        Span {
            start,
            len,
            _buffer: PhantomData,
        }
    }

    fn push(&mut self, src: &[T]) -> Span<T> {
        self.push_with(src.len(), |dst| dst.copy_from_slice(src))
    }
}

impl Arena<f16> {
    fn push_f32(&mut self, src: &[f32]) -> Span<f16> {
        self.push_with(src.len(), |dst| dst.convert_from_f32_slice(src))
    }
}

pub struct HostWeights {
    pub layout: Layout,
    pub f16: Vec<f16>,
    pub f32: Vec<f32>,
}

impl HostWeights {
    pub fn prepare(ck: &Checkpoint) -> Self {
        let (mut h, mut f) = (Arena { data: vec![] }, Arena { data: vec![] });
        let t = &ck.tensors;
        let enc = &t.encoder;
        let tok_embeddings = h.push_f32(ck.f32(&enc.tok_embeddings));
        let emb_norm = f.push(ck.f32(&enc.emb_norm));
        let layers = (enc.layers.iter())
            .map(|l| {
                // With the tensor-core GEMM: permute the q and k rows for RoPE here, and below
                // interleave wi's input and gate rows for GeGLU, once the fragment layout fixes
                // the order. Until then both stay in checkpoint order.
                let wqkv = ck.f32(&l.wqkv);
                let (wqkv, qkv_colsum) = match &l.attn_norm {
                    Some(gamma) => {
                        let folded = fold_layernorm(wqkv, ck.f32(gamma));
                        (h.push(&folded.w), Some(f.push(&folded.colsum)))
                    }
                    None => (h.push_f32(wqkv), None),
                };
                let wi = fold_layernorm(ck.f32(&l.wi), ck.f32(&l.mlp_norm));
                EncoderLayer {
                    wqkv,
                    qkv_colsum,
                    wo: h.push_f32(ck.f32(&l.wo)),
                    wi: h.push(&wi.w),
                    wi_colsum: f.push(&wi.colsum),
                    wo_mlp: h.push_f32(ck.f32(&l.wo_mlp)),
                    global: l.global,
                }
            })
            .collect();
        let final_norm = f.push(ck.f32(&enc.final_norm));
        let type_emb = f.push(ck.f32(&t.type_emb));
        let head = (t.head.iter())
            .map(|l| HeadLayer {
                in_proj: h.push_f32(ck.f32(&l.in_proj_w)),
                in_proj_b: f.push(ck.f32(&l.in_proj_b)),
                out_proj: h.push_f32(ck.f32(&l.out_proj_w)),
                out_proj_b: f.push(ck.f32(&l.out_proj_b)),
                linear1: h.push_f32(ck.f32(&l.linear1_w)),
                linear1_b: f.push(ck.f32(&l.linear1_b)),
                linear2: h.push_f32(ck.f32(&l.linear2_w)),
                linear2_b: f.push(ck.f32(&l.linear2_b)),
                norm1_w: f.push(ck.f32(&l.norm1_w)),
                norm1_b: f.push(ck.f32(&l.norm1_b)),
                norm2_w: f.push(ck.f32(&l.norm2_w)),
                norm2_b: f.push(ck.f32(&l.norm2_b)),
            })
            .collect();
        let s = &t.scorer;
        let scorer = Scorer {
            norm_w: f.push(ck.f32(&s.norm_w)),
            norm_b: f.push(ck.f32(&s.norm_b)),
            lin1: h.push_f32(ck.f32(&s.lin1_w)),
            lin1_b: f.push(ck.f32(&s.lin1_b)),
            lin2: h.push_f32(ck.f32(&s.lin2_w)),
            lin2_b: f.push(ck.f32(&s.lin2_b)),
        };
        Self {
            layout: Layout {
                tok_embeddings,
                emb_norm,
                layers,
                final_norm,
                type_emb,
                head,
                scorer,
            },
            f16: h.data,
            f32: f.data,
        }
    }
}

pub struct GpuWeights {
    pub layout: Layout,
    pub f16: CudaSlice<f16>,
    pub f32: CudaSlice<f32>,
}

impl GpuWeights {
    pub fn upload(stream: &Arc<CudaStream>, host: &HostWeights) -> color_eyre::Result<Self> {
        Ok(Self {
            layout: host.layout.clone(),
            f16: stream.clone_htod(&host.f16)?,
            f32: stream.clone_htod(&host.f32)?,
        })
    }

    /// Downloads both buffers and checks them bit for bit against `host`.
    pub fn verify(&self, stream: &Arc<CudaStream>, host: &HostWeights) -> color_eyre::Result<()> {
        let f16 = stream.clone_dtoh(&self.f16)?;
        let f32 = stream.clone_dtoh(&self.f32)?;
        let bits16 = |v: &[f16]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let bits32 = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        ensure!(
            bits16(&f16) == bits16(&host.f16),
            "the f16 weights differ after the upload"
        );
        ensure!(
            bits32(&f32) == bits32(&host.f32),
            "the f32 weights differ after the upload"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::*;

    /// The checkpoint, from `JULIA_MODEL` or ./model; `None` (test skipped) if it isn't there.
    fn checkpoint() -> Option<Checkpoint> {
        let dir = std::env::var_os("JULIA_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("model"));
        if !dir.join("model.safetensors").exists() {
            eprintln!(
                "skipped: no checkpoint at {} (set JULIA_MODEL)",
                dir.display()
            );
            return None;
        }
        Some(Checkpoint::open(&dir).unwrap())
    }

    fn f16_of(v: &[f32]) -> Vec<u16> {
        v.iter().map(|&x| f16::from_f32(x).to_bits()).collect()
    }

    /// Spot checks that each kind of tensor lands where its span says, aligned, with the right
    /// contents: converted, folded, or copied.
    #[test]
    fn prepared_weights_match_the_checkpoint() {
        let Some(ck) = checkpoint() else { return };
        let w = HostWeights::prepare(&ck);
        let (t, l) = (&ck.tensors, &w.layout);
        let h = |s: Span<f16>| {
            w.f16[s.range()]
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        };
        let f = |s: Span<f32>| &w.f32[s.range()];

        let emb = ck.f32(&t.encoder.tok_embeddings);
        assert_eq!(h(l.tok_embeddings), f16_of(emb));
        assert_eq!(f(l.emb_norm), ck.f32(&t.encoder.emb_norm));
        // Layer 0's wqkv is only converted; the others are folded.
        assert!(l.layers[0].qkv_colsum.is_none());
        assert_eq!(
            h(l.layers[0].wqkv),
            f16_of(ck.f32(&t.encoder.layers[0].wqkv))
        );
        for i in [1, 21] {
            let (src, dst) = (&t.encoder.layers[i], &l.layers[i]);
            let qkv = fold_layernorm(ck.f32(&src.wqkv), ck.f32(src.attn_norm.as_ref().unwrap()));
            assert_eq!(
                h(dst.wqkv),
                qkv.w.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(f(dst.qkv_colsum.unwrap()), qkv.colsum);
            let wi = fold_layernorm(ck.f32(&src.wi), ck.f32(&src.mlp_norm));
            assert_eq!(f(dst.wi_colsum), wi.colsum);
            assert_eq!(h(dst.wo_mlp), f16_of(ck.f32(&src.wo_mlp)));
            assert_eq!(dst.global, src.global);
        }
        assert_eq!(f(l.head[1].linear2_b), ck.f32(&t.head[1].linear2_b));
        assert_eq!(h(l.scorer.lin2), f16_of(ck.f32(&t.scorer.lin2_w)));

        let spans16 = [l.tok_embeddings, l.layers[7].wi, l.scorer.lin2];
        let spans32 = [l.emb_norm, l.layers[7].wi_colsum, l.scorer.lin2_b];
        assert!(
            spans16
                .iter()
                .all(|s| (s.start * 2).is_multiple_of(ALIGN_BYTES))
        );
        assert!(
            spans32
                .iter()
                .all(|s| (s.start * 4).is_multiple_of(ALIGN_BYTES))
        );
        eprintln!(
            "f16: {:.1} MB, f32: {:.1} KB",
            (w.f16.len() * 2) as f64 / 1e6,
            (w.f32.len() * 4) as f64 / 1e3
        );
    }
}
