use super::dataset::GPTItem;
use burn::data::dataloader::batcher::Batcher;
use burn::prelude::{Backend, Int};
use burn::tensor::{Tensor, TensorData};

#[derive(Clone, Debug)]
pub struct GPTBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,  // [batch_size, context_size]
    pub targets: Tensor<B, 2, Int>, // [batch_size, context_size]
}

pub struct GPTBatcher;

impl<B: Backend> Batcher<B, GPTItem, GPTBatch<B>> for GPTBatcher {
    fn batch(&self, items: Vec<GPTItem>, device: &B::Device) -> GPTBatch<B> {
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

        GPTBatch { inputs, targets }
    }
}
