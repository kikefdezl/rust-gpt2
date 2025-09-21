use super::dataset::GptItem;
use burn::data::dataloader::batcher::Batcher;
use burn::prelude::{Backend, Int};
use burn::tensor::{Tensor, TensorData};

#[derive(Clone, Debug)]
pub struct GptBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,  // [batch_size, context_size]
    pub targets: Tensor<B, 2, Int>, // [batch_size, context_size]
}

pub struct GptBatcher;

impl<B: Backend> Batcher<B, GptItem, GptBatch<B>> for GptBatcher {
    fn batch(&self, items: Vec<GptItem>, device: &B::Device) -> GptBatch<B> {
        let inputs = items
            .iter()
            .map(|item| TensorData::new(item.input_ids.clone(), vec![item.input_ids.len()]))
            .map(|data| Tensor::<B, 1, Int>::from_data(data, device))
            .collect();

        let targets = items
            .iter()
            .map(|item| TensorData::new(item.target_ids.clone(), vec![item.target_ids.len()]))
            .map(|data| Tensor::<B, 1, Int>::from_data(data, device))
            .collect();

        let inputs = Tensor::stack::<2>(inputs, 0);
        let targets = Tensor::stack::<2>(targets, 0);

        GptBatch { inputs, targets }
    }
}
