use burn::data::dataset::Dataset;

#[derive(Clone, Debug)]
pub struct GptItem {
    pub input_ids: Vec<u32>,
    pub target_ids: Vec<u32>,
}

pub struct GptDataset {
    pub items: Vec<GptItem>,
}

impl GptDataset {
    pub fn new(token_ids: &[u32], max_length: usize, stride: usize) -> GptDataset {
        let mut items: Vec<GptItem> = Vec::new();

        for i in (0..token_ids.len() - max_length).step_by(stride) {
            let input_ids: Vec<u32> = token_ids.iter().skip(i).take(max_length).cloned().collect();
            let target_ids: Vec<u32> = token_ids
                .iter()
                .skip(i + 1)
                .take(max_length)
                .cloned()
                .collect();

            items.push(GptItem {
                input_ids,
                target_ids,
            });
        }
        GptDataset { items }
    }
}

impl Dataset<GptItem> for GptDataset {
    fn get(&self, index: usize) -> Option<GptItem> {
        Some(self.items[index].clone())
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

impl GptDataset {
    pub fn split_to_train_val(mut self, val_ratio: f32) -> (Self, Self) {
        let n_elements = self.items.len();
        let idx_split = n_elements * (100 - ((val_ratio * 100.0) as usize)) / 100;
        let val_items = self.items.split_off(idx_split);
        (self, GptDataset { items: val_items })
    }
}
