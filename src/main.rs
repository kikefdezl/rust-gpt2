use burn::backend::Candle;
use burn::backend::candle::CandleDevice;
use burn::data::dataloader::{DataLoader, DataLoaderBuilder};
use burn::nn;
use burn::prelude::*;
use burn::tensor::Distribution;
use llm_from_scratch::model::GptConfig124M;
use tiktoken_rs::r50k_base;

use std::fs::read_to_string;
use std::sync::Arc;

use llm_from_scratch::attention::MultiHeadAttention;
use llm_from_scratch::batcher::{GPTBatch, GPTBatcher};
use llm_from_scratch::config;
use llm_from_scratch::dataset::GPTDatasetV1;
use llm_from_scratch::model::DummyGPTModel;

const STRIDE_LEN: usize = 4;
const CONTEXT_LEN: usize = 1000;
const BATCH_SIZE: usize = 8;
const NUM_WORKERS: usize = 4;

fn main() {
    type Backend = Candle;
    let device = CandleDevice::Cpu;

    sandbox::<Backend>(&device);
}

fn sandbox<B: Backend>(device: &B::Device) {
    let text = read_to_string(config::RAW_DATA_FILE).unwrap();

    let tokenizer = r50k_base().unwrap();
    let token_ids = tokenizer.encode_ordinary(&text);
    let token_ids: Vec<u32> = token_ids.into_iter().skip(50).collect();

    let dataset = GPTDatasetV1::new(&token_ids, CONTEXT_LEN, STRIDE_LEN);

    let dataloader: Arc<dyn DataLoader<B, GPTBatch<B>>> = DataLoaderBuilder::new(GPTBatcher)
        .batch_size(BATCH_SIZE)
        .num_workers(NUM_WORKERS)
        .build(dataset);

    let config = GptConfig124M::default();
    let model: DummyGPTModel<B> = config.init(device);

    let first = dataloader.iter().next().unwrap().inputs;

    println!("{}", &first);
    let out = model.forward(first);
    println!("{}", out);
}
