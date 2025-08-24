use burn::backend::Candle;
use burn::backend::candle::CandleDevice;
use burn::data::dataloader::{DataLoader, DataLoaderBuilder};
use burn::prelude::*;
use burn::tensor::cast::ToElement;
use llm_from_scratch::model::GptConfig124M;
use std::sync::Arc;
use tiktoken_rs::r50k_base;

use std::fs::read_to_string;

use llm_from_scratch::batcher::{GPTBatch, GPTBatcher};
use llm_from_scratch::config;
use llm_from_scratch::dataset::GPTDatasetV1;
use llm_from_scratch::model::GPTModel;

const STRIDE_LEN: usize = 4;
const CONTEXT_LEN: usize = 10;
const BATCH_SIZE: usize = 2;
const NUM_WORKERS: usize = 4;

fn main() {
    type Backend = Candle;
    let device = CandleDevice::Cpu;

    generate_text::<Backend>(&device);
    sandbox::<Backend>(&device);
}

fn generate_text<B: Backend>(device: &B::Device) {
    let text = String::from("Hello I am ");

    let tokenizer = r50k_base().unwrap();
    let token_ids = tokenizer.encode_ordinary(&text);

    let config = GptConfig124M::default();
    let model: GPTModel<B> = config.init(device);

    let input_data = TensorData::new(token_ids.clone(), vec![token_ids.len()]);
    let input: Tensor<B, 1, Int> = Tensor::from_data(input_data, device);
    let mut input: Tensor<B, 2, Int> = input.unsqueeze();
    print!("{text}");
    for _ in 0..100 {
        let last_idx = input.clone().dims()[1];
        let first_idx = last_idx.saturating_sub(CONTEXT_LEN);

        input = input.clone().slice_dim(1, first_idx..last_idx);

        let logits = model.forward(input.clone());
        let last_logits = logits
            .clone()
            .select(1, Tensor::from_ints([logits.dims()[1] - 1], device));

        let probas = burn::tensor::activation::softmax(last_logits, 2);
        let idx_next: Tensor<B, 2, Int> = probas.argmax(2).squeeze(1);

        let decoded = tokenizer
            .decode(vec![idx_next.clone().into_scalar().to_u32()])
            .unwrap_or(String::from(""));
        print!("{decoded}");

        input = Tensor::cat(vec![input, idx_next], 1);
    }
}

fn sandbox<B: Backend>(device: &B::Device) {
    let text = read_to_string(config::RAW_DATA_FILE).unwrap();

    let tokenizer = r50k_base().unwrap();
    let token_ids = tokenizer.encode_ordinary(&text);

    let dataset = GPTDatasetV1::new(&token_ids, CONTEXT_LEN, STRIDE_LEN);
    let dataloader: Arc<dyn DataLoader<B, GPTBatch<B>>> = DataLoaderBuilder::new(GPTBatcher)
        .batch_size(BATCH_SIZE)
        .num_workers(NUM_WORKERS)
        .build(dataset);

    let config = GptConfig124M::default();
    let model: GPTModel<B> = config.init(device);

    let first = dataloader.iter().next().unwrap().inputs;

    println!("{}", &first);
    let out = model.forward(first);
    println!("{}", out);
}
