use burn::data::dataset::Dataset;

#[derive(Clone, Debug)]
pub struct GPTItem {
    pub input_ids: Vec<u32>,
    pub target_ids: Vec<u32>,
}

pub struct GPTDatasetV1 {
    pub items: Vec<GPTItem>,
}

impl GPTDatasetV1 {
    pub fn new(token_ids: &[u32], max_length: usize, stride: usize) -> GPTDatasetV1 {
        let mut items: Vec<GPTItem> = Vec::new();

        for i in (0..token_ids.len() - max_length).step_by(stride) {
            let input_ids: Vec<u32> = token_ids.iter().skip(i).take(max_length).cloned().collect();
            let target_ids: Vec<u32> = token_ids
                .iter()
                .skip(i + 1)
                .take(max_length)
                .cloned()
                .collect();

            items.push(GPTItem {
                input_ids,
                target_ids,
            });
        }
        GPTDatasetV1 { items }
    }
}

impl Dataset<GPTItem> for GPTDatasetV1 {
    fn get(&self, index: usize) -> Option<GPTItem> {
        Some(self.items[index].clone())
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

impl GPTDatasetV1 {
    pub fn split_to_train_val(mut self, val_ratio: f32) -> (Self, Self) {
        let n_elements = self.items.len();
        let idx_split = n_elements * (100 - ((val_ratio * 100.0) as usize)) / 100;
        let val_items = self.items.split_off(idx_split);
        (self, GPTDatasetV1 { items: val_items })
    }
}
