use bpe_openai::cl100k_base;
use burn::backend::Candle;
use burn::backend::candle::CandleDevice;
use burn::data::dataloader::{DataLoader, DataLoaderBuilder};
use burn::nn::{self, Embedding};
use burn::prelude::*;
use burn::tensor::Distribution;
use llm_from_scratch::batcher::{GPTBatch, GPTBatcher};
use llm_from_scratch::config;
use llm_from_scratch::dataset::GPTDatasetV1;
use std::sync::Arc;

use std::fs::read_to_string;

const STRIDE_LEN: usize = 4;
const CONTEXT_LEN: usize = 4;
const BATCH_SIZE: usize = 8;
const NUM_WORKERS: usize = 4;
const EMBEDDING_DIM: usize = 256;

type Backend = Candle;

const DEVICE: CandleDevice = CandleDevice::Cpu;

fn main() {
    let embeddings: Tensor<Backend, 2> = Tensor::from_data(
        [
            [0.43, 0.15, 0.89],
            [0.55, 0.87, 0.66],
            [0.57, 0.85, 0.64],
            [0.22, 0.58, 0.33],
            [0.77, 0.25, 0.10],
            [0.05, 0.80, 0.55],
        ],
        &DEVICE,
    );
    let (d_in, d_out) = (3, 2);

    let distribution = Distribution::Uniform(0.0, 1.0);
    let w_query: Tensor<Backend, 2> = Tensor::random([d_in, d_out], distribution, &DEVICE);
    let w_query = burn::module::Param::from_tensor(w_query).set_require_grad(false);

    let q = embeddings.clone().matmul(w_query.val());
    println!("{q}");
    // W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    // W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
}

fn _simple_attention() {
    let embeddings: Tensor<Backend, 2> = Tensor::from_data(
        [
            [0.43, 0.15, 0.89],
            [0.55, 0.87, 0.66],
            [0.57, 0.85, 0.64],
            [0.22, 0.58, 0.33],
            [0.77, 0.25, 0.10],
            [0.05, 0.80, 0.55],
        ],
        &DEVICE,
    );

    let attn_weights = embeddings.clone().matmul(embeddings.clone().transpose());
    let attn_weights = burn::tensor::activation::softmax(attn_weights, 1);
    let result = attn_weights.matmul(embeddings);
    println!("{}", &result);
}

fn _preprocess() {
    let text = read_to_string(config::RAW_DATA_FILE).unwrap();

    let tokenizer = cl100k_base();
    let token_ids = tokenizer.encode(&text);
    let token_ids: Vec<u32> = token_ids.into_iter().skip(50).collect();

    let dataset = GPTDatasetV1::new(&token_ids, CONTEXT_LEN, STRIDE_LEN);

    let dataloader: Arc<dyn DataLoader<Backend, GPTBatch<Backend>>> =
        DataLoaderBuilder::new(GPTBatcher)
            .batch_size(BATCH_SIZE)
            .num_workers(NUM_WORKERS)
            .build(dataset);

    let embedding_layer: nn::Embedding<Backend> =
        nn::EmbeddingConfig::new(100_000, EMBEDDING_DIM).init(&DEVICE);
    let pos_embedding_layer: nn::Embedding<Backend> =
        nn::EmbeddingConfig::new(CONTEXT_LEN, EMBEDDING_DIM).init(&DEVICE);

    let first = dataloader.iter().next().unwrap().inputs;

    let token_embeddings = embedding_layer.forward(first);
    let pos_embeddings = pos_embedding_layer
        .forward(Tensor::arange(0..CONTEXT_LEN as i64, &DEVICE).reshape([1, CONTEXT_LEN]));

    let input_embeddings = token_embeddings + pos_embeddings;

    println!("{}", embedding_layer.weight.val());
    println!("{}", pos_embedding_layer.weight.val());
    println!("{:?}", input_embeddings.shape());
}
