use burn::data::dataloader::DataLoaderBuilder;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::AdamWConfig;
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::{AccuracyMetric, LossMetric};
use burn::train::{ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep};
use tiktoken_rs::r50k_base;

use std::fs::read_to_string;

use crate::data::batcher::{GptBatch, GptBatcher};
use crate::data::dataset::GptDataset;
use crate::model::gpt2::Gpt2;
use crate::model::gpt2::Gpt2Config;

#[derive(Config)]
pub struct TrainConfig {
    model: Gpt2Config,
    optimizer: AdamWConfig,
    #[config(default = 2)]
    batch_size: usize,
    #[config(default = 1.0e-4)]
    learning_rate: f64,
    #[config(default = 10)]
    num_epochs: usize,
    #[config(default = 2)]
    num_workers: usize,
    #[config(default = 42)]
    seed: u64,
    #[config(default = 256)]
    stride_length: usize,
    #[config(default = 0.1)]
    val_ratio: f32,
}

pub fn train<B: AutodiffBackend>(
    dataset: &str,
    workdir: &str,
    config: &TrainConfig,
    device: &B::Device,
) {
    config
        .save(format!("{workdir}/config.json"))
        .expect("Config should be saved successfully");

    let text = read_to_string(dataset).unwrap();

    let tokenizer = r50k_base().unwrap();
    let token_ids = tokenizer.encode_ordinary(&text);

    let dataset = GptDataset::new(
        &token_ids,
        config.model.context_length,
        config.stride_length,
    );
    let (train_dataset, val_dataset) = dataset.split_to_train_val(config.val_ratio);

    let train_dataloader = DataLoaderBuilder::new(GptBatcher)
        .num_workers(config.num_workers)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .build(train_dataset);
    let val_dataloader = DataLoaderBuilder::new(GptBatcher)
        .num_workers(config.num_workers)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .build(val_dataset);

    let model: Gpt2<B> = config.model.init(device);

    let learner = LearnerBuilder::new(workdir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.num_epochs)
        .summary()
        .build(model, config.optimizer.init(), config.learning_rate);

    let model_trained = learner.fit(train_dataloader, val_dataloader);

    model_trained
        .save_file(format!("{workdir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}

impl<B: Backend> Gpt2<B> {
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

impl<B: AutodiffBackend> TrainStep<GptBatch<B>, ClassificationOutput<B>> for Gpt2<B> {
    fn step(&self, batch: GptBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.inputs, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<GptBatch<B>, ClassificationOutput<B>> for Gpt2<B> {
    fn step(&self, batch: GptBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.inputs, batch.targets)
    }
}
