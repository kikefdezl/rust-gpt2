use burn::data::dataloader::{DataLoader, DataLoaderBuilder};
use burn::nn::loss::{CrossEntropyLoss, CrossEntropyLossConfig};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use std::sync::Arc;
use tiktoken_rs::r50k_base;

use std::fs::read_to_string;

use crate::batcher::{GPTBatch, GPTBatcher};
use crate::config;
use crate::dataset::GPTDatasetV1;
use crate::model::GPTModel;
use crate::model::GptConfig124M;
use crate::tokenizer::token_ids_to_text;

pub struct TrainConfig {
    batch_size: usize,
    context_length: usize,
    stride_length: usize,
    workers: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            batch_size: 2,
            context_length: 3,
            stride_length: 1,
            workers: 8,
        }
    }
}

impl TrainConfig {
    pub fn with_context_length(mut self, new_length: usize) -> Self {
        self.context_length = new_length;
        self
    }
}

pub fn train<B: AutodiffBackend>(config: &TrainConfig, device: &B::Device) {
    let text = read_to_string(config::RAW_DATA_FILE).unwrap();

    let tokenizer = r50k_base().unwrap();
    let token_ids = tokenizer.encode_ordinary(&text);

    let model_config = GptConfig124M::default().with_context_len(config.context_length);
    let model: GPTModel<B> = model_config.init(device);

    let dataset = GPTDatasetV1::new(&token_ids, config.context_length, config.stride_length);
    let dataloader: Arc<dyn DataLoader<B, GPTBatch<B>>> = DataLoaderBuilder::new(GPTBatcher)
        .batch_size(config.batch_size)
        .num_workers(config.workers)
        .build(dataset);

    let criterion: CrossEntropyLoss<B> = CrossEntropyLossConfig {
        pad_tokens: None,
        weights: None,
        smoothing: None,
        logits: true,
    }
    .init(device);

    for batch in dataloader.iter() {
        let logits: Tensor<B, 3> = model.forward(batch.inputs);
        // let probas = burn::tensor::activation::softmax(logits, 2);
        let logits_flat: Tensor<B, 2> = logits.flatten(0, 1);
        let targets_flat: Tensor<B, 1, Int> = batch.targets.flatten(0, 1);

        let loss = criterion.forward(logits_flat, targets_flat);
        println!("loss {}", &loss);
    }
}
