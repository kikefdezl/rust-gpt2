use burn::backend::Candle;
use burn::backend::candle::CandleDevice;
use burn::data::dataloader::{DataLoader, DataLoaderBuilder};
use burn::prelude::*;
use llm_from_scratch::model::GptConfig124M;
use std::sync::Arc;
use tiktoken_rs::r50k_base;

use std::fs::read_to_string;

use llm_from_scratch::batcher::{GPTBatch, GPTBatcher};
use llm_from_scratch::config;
use llm_from_scratch::dataset::GPTDatasetV1;
use llm_from_scratch::model::GPTModel;
use llm_from_scratch::tokenizer::{text_to_token_ids, token_ids_to_text};

const STRIDE_LEN: usize = 4;
const CONTEXT_LEN: usize = 10;
const BATCH_SIZE: usize = 2;
const NUM_WORKERS: usize = 4;

fn main() {
    type Backend = Candle;
    let device = CandleDevice::Cpu;

    sandbox::<Backend>(&device);
    sandbox_2::<Backend>(&device);
}

fn sandbox<B: Backend>(device: &B::Device) {
    let tokenizer = r50k_base().unwrap();

    let config = GptConfig124M::default().with_context_len(256);
    let model: GPTModel<B> = config.init(device);

    let text = String::from("Hello I am ");
    let token_ids: Tensor<B, 2, Int> = text_to_token_ids(&text, &tokenizer, device).unsqueeze();

    let idx = generate_text_simple(&model, token_ids, 10, config.context_length);

    let decoded = token_ids_to_text(idx.squeeze(0), &tokenizer);
    print!("{decoded}");
}

fn generate_text_simple<B: Backend>(
    model: &GPTModel<B>,
    mut token_ids: Tensor<B, 2, Int>,
    max_new_tokens: usize,
    context_size: usize,
) -> Tensor<B, 2, Int> {
    for _ in 0..max_new_tokens {
        let last_idx = token_ids.dims()[1];
        let first_idx = last_idx.saturating_sub(context_size);

        token_ids = token_ids.slice_dim(1, first_idx..last_idx);

        let logits = model.forward(token_ids.clone()); // size (Batch, Ctx, Vocab)
        let last_idx = logits.dims()[1] - 1;
        let device = &logits.device();
        let last_logits: Tensor<B, 3> = logits.select(1, Tensor::from_ints([last_idx], device));

        let probas: Tensor<B, 2> = burn::tensor::activation::softmax(last_logits.squeeze(0), 1);
        let idx_next: Tensor<B, 1, Int> = probas.argmax(1).squeeze(0);

        token_ids = Tensor::cat(vec![token_ids, idx_next.unsqueeze()], 1);
    }
    token_ids
}

fn sandbox_2<B: Backend>(device: &B::Device) {
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

    // println!("{}", &first);
    let out = model.forward(first);
    // println!("{}", out);
}
