use burn_import::safetensors::LoadArgs;
use std::path::PathBuf;
use burn::prelude::*;

use crate::model::gpt2::{Gpt2, Gpt2Config};

pub fn load_gpt2<B: Backend>(weights_path: PathBuf, device: &B::Device) -> Gpt2<B>{
    let load_args = LoadArgs::new(weights_path);
        // Map *.downsample.0.* -> *.downsample.conv.*
        // .with_key_remap("(.+)\.downsample\.0\.(.+)", "$1.downsample.conv.$2")
        // // Map *.downsample.1.* -> *.downsample.bn.*
        // .with_key_remap("(.+)\.downsample\.1\.(.+)", "$1.downsample.bn.$2");


    let model_config = Gpt2Config::new();
    model_config.init(device)
}

