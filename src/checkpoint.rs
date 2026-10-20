//! The checkpoint files, read with julia-rs's parsers.
//! An alternative implementation of `julia_rs::load_model` that is
//! suitable for the GPU.

use std::{fs::File, path::Path};

use color_eyre::eyre::{Context, bail, ensure};
use julia_rs::{
    config::ModelConfig,
    safetensors::{Tensor, Weights, parse_header},
    tokenizer::Tokenizer,
};
use memmap2::Mmap;

pub struct Checkpoint {
    map: Mmap,
    /// Offset of the safetensors data section within `map`.
    data_start: usize,
    pub config: ModelConfig,
    /// Where each tensor is in the data section (shapes checked against `config`).
    pub tensors: Weights,
    #[expect(dead_code)] // read by the sequence builder, next
    pub tokenizer: Tokenizer,
}

impl Checkpoint {
    /// Opens a checkpoint directory laid out like `model/`.
    pub fn open(dir: &Path) -> color_eyre::Result<Self> {
        let path = |p: &str| dir.join(p);
        let config = ModelConfig::load(&path("encoder/config.json"), &path("julia_config.json"))?;

        let safetensors = path("model.safetensors");
        let file = File::open(&safetensors).wrap_err_with(|| safetensors.display().to_string())?;
        // SAFETY: the file is only read, and not modified while the program runs.
        let map = unsafe { Mmap::map(&file)? };
        let Some(len_bytes) = map.get(0..8) else {
            bail!("model.safetensors is too short");
        };
        let len = u64::from_le_bytes(len_bytes.try_into().unwrap()) as usize;
        let data_start = 8usize.saturating_add(len);
        let Some(header) = map.get(8..data_start) else {
            bail!("model.safetensors header length {len} exceeds file size");
        };
        let tensors = parse_header(header, map.len() - data_start, &config)?;

        let tokenizer = Tokenizer::load(
            &path("tokenizer/tokenizer.json"),
            &path("tokenizer/tokenizer_config.json"),
        )?;
        ensure!(
            tokenizer.vocab_size == config.vocab,
            "tokenizer has {} ids but encoder config says vocab_size is {}",
            tokenizer.vocab_size,
            config.vocab
        );
        Ok(Self {
            map,
            data_start,
            config,
            tensors,
            tokenizer,
        })
    }

    pub fn f32(&self, t: &Tensor) -> &[f32] {
        t.as_f32(&self.map[self.data_start..])
    }
}
