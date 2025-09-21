use burn::backend::Autodiff;
use burn::backend::candle::{Candle, CandleDevice};
use burn::optim::AdamWConfig;
use std::fs::{create_dir, exists};

use rust_gpt2::model::gpt2::Gpt2Config;
use rust_gpt2::train::{TrainConfig, train};

const RAW_DATA_FILE: &str = "data/raw/the-verdict.txt";
const TRAINING_WORKDIR: &str = "runs/";

fn main() {
    type Backend = Candle;
    type AutodiffBackend = Autodiff<Backend>;
    let device = CandleDevice::Cpu;

    let context_length = 256;
    let model_config = Gpt2Config::new().with_context_length(context_length);
    let train_config =
        TrainConfig::new(model_config, AdamWConfig::new()).with_stride_length(context_length);

    if !exists(TRAINING_WORKDIR).unwrap_or(false) {
        create_dir(TRAINING_WORKDIR).expect("Workdir creation should not panic.");
    }

    train::<AutodiffBackend>(RAW_DATA_FILE, TRAINING_WORKDIR, &train_config, &device);
}
