use super::dataset::GPTItem;
use burn::data::dataloader::batcher::Batcher;
use burn::prelude::{Backend, Int};
use burn::tensor::{Tensor, TensorData};

#[derive(Clone, Debug)]
pub struct GPTBatch<B: Backend> {
    pub inputs: Tensor<B, 1, Int>,
    pub targets: Tensor<B, 1, Int>,
}

pub struct GPTBatcher;

impl<B: Backend> Batcher<B, GPTItem, GPTBatch<B>> for GPTBatcher {
    fn batch(&self, items: Vec<GPTItem>, device: &B::Device) -> GPTBatch<B> {
        let inputs = items
            .iter()
            .map(|item| Tensor::from_ints(item.input_ids, &device))
            .collect();

        let inputs = Tensor::cat(inputs, 0);
        GPTBatch { inputs, targets }
    }
}
