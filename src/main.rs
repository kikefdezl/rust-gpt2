use bpe_openai::cl100k_base;
use burn::backend::Candle;
use burn::backend::candle::CandleDevice;
use burn::data::dataloader::{DataLoader, DataLoaderBuilder};
use burn::nn;
use burn::prelude::*;
use burn::tensor::Distribution;
use llm_from_scratch::batcher::{GPTBatch, GPTBatcher};
use llm_from_scratch::config;
use llm_from_scratch::dataset::GPTDatasetV1;
use std::sync::Arc;

use std::fs::read_to_string;

use llm_from_scratch::model::SelfAttentionV1;

const STRIDE_LEN: usize = 4;
const CONTEXT_LEN: usize = 4;
const BATCH_SIZE: usize = 8;
const NUM_WORKERS: usize = 4;
const EMBEDDING_DIM: usize = 256;

fn main() {
    type Backend = Candle;
    let device = CandleDevice::Cpu;

    sandbox::<Backend>(&device);
}

fn sandbox<B: Backend>(device: &B::Device) {
    let embeddings: Tensor<B, 2> = Tensor::from_data(
        [
            [0.43, 0.15, 0.89],
            [0.55, 0.87, 0.66],
            [0.57, 0.85, 0.64],
            [0.22, 0.58, 0.33],
            [0.77, 0.25, 0.10],
            [0.05, 0.80, 0.55],
        ],
        device,
    );

    let attn: SelfAttentionV1<B> = SelfAttentionV1::new(3, 2, device);
    let context_vector = attn.forward(embeddings);
    println!("{}", context_vector);
}

fn _preprocess<B: Backend>(device: &B::Device) {
    let text = read_to_string(config::RAW_DATA_FILE).unwrap();

    let tokenizer = cl100k_base();
    let token_ids = tokenizer.encode(&text);
    let token_ids: Vec<u32> = token_ids.into_iter().skip(50).collect();

    let dataset = GPTDatasetV1::new(&token_ids, CONTEXT_LEN, STRIDE_LEN);

    let dataloader: Arc<dyn DataLoader<B, GPTBatch<B>>> = DataLoaderBuilder::new(GPTBatcher)
        .batch_size(BATCH_SIZE)
        .num_workers(NUM_WORKERS)
        .build(dataset);

    let embedding_layer: nn::Embedding<B> =
        nn::EmbeddingConfig::new(100_000, EMBEDDING_DIM).init(device);
    let pos_embedding_layer: nn::Embedding<B> =
        nn::EmbeddingConfig::new(CONTEXT_LEN, EMBEDDING_DIM).init(device);

    let first = dataloader.iter().next().unwrap().inputs;

    let token_embeddings = embedding_layer.forward(first);
    let pos_embeddings = pos_embedding_layer
        .forward(Tensor::arange(0..CONTEXT_LEN as i64, device).reshape([1, CONTEXT_LEN]));

    let input_embeddings = token_embeddings + pos_embeddings;

    println!("{}", embedding_layer.weight.val());
    println!("{}", pos_embedding_layer.weight.val());
    println!("{:?}", input_embeddings.shape());
}
