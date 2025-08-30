use burn::data::dataloader::{DataLoader, DataLoaderBuilder};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep};
use std::sync::Arc;
use tiktoken_rs::r50k_base;

use std::fs::read_to_string;

use crate::batcher::{GPTBatch, GPTBatcher};
use crate::config;
use crate::dataset::GPTDatasetV1;
use crate::model::GPTModel;
use crate::model::GptConfig;
use crate::tokenizer::token_ids_to_text;

#[derive(Config)]
pub struct TrainConfig {
    model: GptConfig,
    #[config(default = 8)]
    batch_size: usize,
    #[config(default = 1.0e-4)]
    learning_rate: f64,
    #[config(default = 10)]
    num_epochs: usize,
    #[config(default = 8)]
    num_workers: usize,
    #[config(default = 6)]
    stride_length: usize,
    #[config(default = 0.1)]
    val_ratio: f32,
}

pub fn train<B: AutodiffBackend>(config: &TrainConfig, device: &B::Device) {
    let text = read_to_string(config::RAW_DATA_FILE).unwrap();

    let tokenizer = r50k_base().unwrap();
    let token_ids = tokenizer.encode_ordinary(&text);

    let model: GPTModel<B> = config.model.init(device);

    let (train_ids, val_ids) = split_to_train_val(token_ids, config.val_ratio);

    let train_dataset = GPTDatasetV1::new(
        &train_ids,
        config.model.context_length,
        config.stride_length,
    );
    let val_dataset =
        GPTDatasetV1::new(&val_ids, config.model.context_length, config.stride_length);

    let train_dataloader: Arc<dyn DataLoader<B, GPTBatch<B>>> = DataLoaderBuilder::new(GPTBatcher)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(train_dataset);
    let val_dataloader: Arc<dyn DataLoader<B, GPTBatch<B>>> = DataLoaderBuilder::new(GPTBatcher)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(val_dataset);

    for batch in train_dataloader.iter() {
        let out = model.forward_classification(batch.inputs, batch.targets);
        // println!("loss {:?}", out);
    }
}

fn split_to_train_val<T>(mut vec: Vec<T>, val_ratio: f32) -> (Vec<T>, Vec<T>) {
    let n_elements = vec.len();
    let idx_split = n_elements * (100 - ((val_ratio * 100.0) as usize)) / 100;
    let val = vec.split_off(idx_split);
    (vec, val)
}

impl<B: Backend> GPTModel<B> {
    pub fn forward_classification(
        &self,
        inputs: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
    ) -> ClassificationOutput<B> {
        let logits: Tensor<B, 3> = self.forward(inputs);
        let logits_flat: Tensor<B, 2> = logits.flatten(0, 1);
        let targets_flat: Tensor<B, 1, Int> = targets.flatten(0, 1);

        let loss = CrossEntropyLossConfig::new()
            .init(&targets_flat.device())
            .forward(logits_flat.clone(), targets_flat.clone());

        ClassificationOutput::new(loss, logits_flat, targets_flat)
    }
}

impl<B: AutodiffBackend> TrainStep<GPTBatch<B>, ClassificationOutput<B>> for GPTModel<B> {
    fn step(&self, batch: GPTBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.inputs, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<GPTBatch<B>, ClassificationOutput<B>> for GPTModel<B> {
    fn step(&self, batch: GPTBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.inputs, batch.targets)
    }
}

// fn optimize<B, O>(self, optim: &mut O, lr: f64, grads: GradientsParams) -> Self
