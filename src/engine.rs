//! A loaded model, for programs that embed julia-cuda: the kernels, the weights on the GPU and the
//! split-K scratch, answering requests on one stream. `launch` queues one request's forward pass
//! without waiting for it, so `Engine::run` queues a batch's requests back to back and waits once
//! (the CLI's `serve` runs them one at a time).

use std::{path::Path, sync::Arc};

use color_eyre::eyre::ensure;
use cudarc::driver::{
    CudaContext, CudaEvent, CudaStream,
    sys::{CUdevice_attribute, CUevent_flags},
};
use julia_rs::sequence::{Request, Sequence, sequence};

use crate::{
    D, Kernels,
    checkpoint::Checkpoint,
    embed_into,
    forward::{GpuEncoder, GpuHead},
    gemm, load_kernels,
    weights::{GpuWeights, HostWeights},
};

/// Requests `Engine::run` queues before it waits: each holds its buffers until it has run (about
/// 15 MB at 1024 tokens).
const QUEUE: usize = 16;

/// The model on GPU 0.
pub struct Engine {
    pub stream: Arc<CudaStream>,
    pub kernels: Kernels,
    pub gpu: GpuWeights,
    pub ck: Checkpoint,
    pub scratch: gemm::Scratch,
    pub sms: usize,
    pub max_length: usize,
    pub head_length: usize,
    /// The GPU's name.
    pub device: String,
}

/// One request's forward pass, queued: its buffers (alive until it has run) and its timing events.
pub struct Pending<'a> {
    _enc: GpuEncoder<'a>,
    head: GpuHead<'a>,
    start: CudaEvent,
    end: CudaEvent,
}

impl Pending<'_> {
    /// Waits for the forward pass: the logits, and its GPU time in ms (embed to logits).
    pub fn finish(self) -> color_eyre::Result<(Vec<f32>, f32)> {
        let logits = self.head.stream.clone_dtoh(&self.head.logits)?;
        let ms = self.start.elapsed_ms(&self.end)?;
        ensure!(
            logits.iter().all(|z| z.is_finite()),
            "inference returned nonfinite logits"
        );
        Ok((logits, ms))
    }
}

/// Queues `seq`'s forward pass (embed, the encoder, the head) on `stream`: 124 launches, or 36
/// with `layer_fused`.
#[allow(clippy::too_many_arguments)]
pub fn launch<'a>(
    stream: &'a Arc<CudaStream>,
    kernels: &'a Kernels,
    scratch: &'a gemm::Scratch,
    gpu: &'a GpuWeights,
    ck: &Checkpoint,
    sms: usize,
    seq: &Sequence,
    layer_fused: bool,
) -> color_eyre::Result<Pending<'a>> {
    let c = &ck.config;
    let n = seq.ids.len();
    ensure!(
        seq.markers.iter().all(|&m| m < n),
        "a marker is past the sequence"
    );
    let markers: Vec<u32> = seq.markers.iter().map(|&m| m as u32).collect();
    let ids = stream.clone_htod(&seq.ids)?;
    let mut enc = GpuEncoder::new(
        stream,
        kernels,
        scratch,
        gpu,
        c.norm_eps,
        c.local_window / 2,
        sms,
        stream.alloc_zeros(n * D)?,
        c.intermediate,
    )?;
    enc.fused = layer_fused;
    let mut head = GpuHead::new(stream, kernels, scratch, gpu, c.norm_eps, sms, n, &markers)?;

    let timing = Some(CUevent_flags::CU_EVENT_DEFAULT);
    let start = stream.record_event(timing)?;
    embed_into(stream, kernels, gpu, &ids, c.norm_eps, &mut enc.x)?;
    for l in 0..gpu.layout.layers.len() {
        enc.launch_layer(l)?;
    }
    head.launch_all(&enc.x, seq.qtype as usize)?;
    let end = stream.record_event(timing)?;
    Ok(Pending {
        _enc: enc,
        head,
        start,
        end,
    })
}

impl Engine {
    /// Whether this machine can run it: the CUDA driver library loads and sees a GPU.
    pub fn available() -> bool {
        // SAFETY: only tries to load the library (no CUDA call before it's known to be there).
        let present = unsafe { cudarc::driver::sys::is_culib_present() };
        present && CudaContext::device_count().is_ok_and(|n| n > 0)
    }

    /// Opens the checkpoint in `dir` and uploads it to GPU 0. `max_length` defaults to the
    /// model's maximum; the requests' tokenization takes both lengths, as julia-rs's.
    pub fn load(
        dir: &Path,
        max_length: Option<usize>,
        head_length: usize,
    ) -> color_eyre::Result<Self> {
        let ctx = CudaContext::new(0)?;
        let kernels = load_kernels(&ctx)?;
        let ck = Checkpoint::open(dir)?;
        let c = &ck.config;
        ensure!(
            c.hidden == D,
            "the kernels are built for hidden size {D}, the model's is {}",
            c.hidden
        );
        let max_length = max_length.unwrap_or(c.max_positions);
        ensure!(
            max_length <= c.max_positions,
            "max_length {max_length} is over the model's {}",
            c.max_positions
        );
        let host = HostWeights::prepare(&ck);
        let stream = ctx.default_stream();
        let gpu = GpuWeights::upload(&stream, &host)?;
        let scratch = gemm::Scratch::new(&stream, max_length)?;
        stream.synchronize()?;
        let sms = ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
        Ok(Self {
            stream,
            kernels,
            gpu,
            ck,
            scratch,
            sms: sms as usize,
            max_length,
            head_length,
            device: ctx.name()?,
        })
    }

    /// `req` tokenized for the model (on the CPU; any thread).
    pub fn sequence(&self, req: &Request) -> color_eyre::Result<Sequence> {
        sequence(&self.ck.tokenizer, req, self.max_length, self.head_length)
    }

    /// Runs `seqs` (each from `sequence`) in order: per request, the logits and the GPU time in
    /// ms. Up to `QUEUE` requests are queued before waiting, so the GPU doesn't idle between them.
    pub fn run(
        &self,
        seqs: &[Sequence],
        layer_fused: bool,
    ) -> color_eyre::Result<Vec<(Vec<f32>, f32)>> {
        let mut out = Vec::with_capacity(seqs.len());
        for chunk in seqs.chunks(QUEUE) {
            let pending = (chunk.iter())
                .map(|seq| {
                    launch(
                        &self.stream,
                        &self.kernels,
                        &self.scratch,
                        &self.gpu,
                        &self.ck,
                        self.sms,
                        seq,
                        layer_fused,
                    )
                })
                .collect::<color_eyre::Result<Vec<_>>>()?;
            for p in pending {
                out.push(p.finish()?);
            }
        }
        Ok(out)
    }
}
