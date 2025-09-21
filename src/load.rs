use burn::prelude::*;
use std::path::PathBuf;

use burn::record::{FullPrecisionSettings, Recorder};
use burn_import::safetensors::{AdapterType, LoadArgs, SafetensorsFileRecorder};

use crate::model::gpt2::Gpt2Record;
use crate::model::gpt2::{Gpt2, Gpt2Config};

pub fn load_gpt2<B: Backend>(weights_path: PathBuf, device: &B::Device) -> Gpt2<B> {
    let load_args = LoadArgs::new(weights_path)
        .with_adapter_type(AdapterType::NoAdapter)
        // input
        .with_key_remap(r"^wte$", "token_embedding.weight")
        .with_key_remap(r"^wpe$", "positional_embedding.weight")
        // attention blocks
        .with_key_remap(
            r"blocks\.(\d+)\.attn\.c_attn\.w_q",
            "transformer_blocks.$1.mha.w_query.weight",
        )
        .with_key_remap(
            r"blocks\.(\d+)\.attn\.c_attn\.b_q",
            "transformer_blocks.$1.mha.w_query.bias",
        )
        .with_key_remap(
            r"blocks\.(\d+)\.attn\.c_attn\.w_k",
            "transformer_blocks.$1.mha.w_key.weight",
        )
        .with_key_remap(
            r"blocks\.(\d+)\.attn\.c_attn\.b_k",
            "transformer_blocks.$1.mha.w_key.bias",
        )
        .with_key_remap(
            r"blocks\.(\d+)\.attn\.c_attn\.w_v",
            "transformer_blocks.$1.mha.w_value.weight",
        )
        .with_key_remap(
            r"blocks\.(\d+)\.attn\.c_attn\.b_v",
            "transformer_blocks.$1.mha.w_value.bias",
        )
        .with_key_remap(
            r"blocks\.(\d+)\.attn\.c_proj\.w",
            "transformer_blocks.$1.mha.out_proj.weight",
        )
        .with_key_remap(
            r"blocks\.(\d+)\.attn\.c_proj\.b",
            "transformer_blocks.$1.mha.out_proj.bias",
        )
        // feed forward blocks
        .with_key_remap(
            r"blocks\.(\d+)\.mlp\.c_fc\.w",
            "transformer_blocks.$1.ff.pre.weight",
        )
        .with_key_remap(
            r"blocks\.(\d+)\.mlp\.c_fc\.b",
            "transformer_blocks.$1.ff.pre.bias",
        )
        .with_key_remap(
            r"blocks\.(\d+)\.mlp\.c_proj\.w",
            "transformer_blocks.$1.ff.post.weight",
        )
        .with_key_remap(
            r"blocks\.(\d+)\.mlp\.c_proj\.b",
            "transformer_blocks.$1.ff.post.bias",
        )
        // norms
        .with_key_remap(
            r"blocks\.(\d+)\.ln_1\.g",
            "transformer_blocks.$1.norm1.scale",
        )
        .with_key_remap(
            r"blocks\.(\d+)\.ln_1\.b",
            "transformer_blocks.$1.norm1.shift",
        )
        .with_key_remap(
            r"blocks\.(\d+)\.ln_2\.g",
            "transformer_blocks.$1.norm2.scale",
        )
        .with_key_remap(
            r"blocks\.(\d+)\.ln_2\.b",
            "transformer_blocks.$1.norm2.shift",
        )
        // final layers
        .with_key_remap(r"^g$", "norm.scale")
        .with_key_remap(r"^b$", "norm.shift")
        .with_key_remap(r"^wte_out$", "linear_out.weight");

    let model_config = Gpt2Config::new();

    let record: Gpt2Record<B> = SafetensorsFileRecorder::<FullPrecisionSettings>::default()
        .load(load_args, device)
        .expect("Should load Safetensors model weights");

    model_config.init(device).load_record(record)
}
