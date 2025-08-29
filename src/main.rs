use burn::backend::candle::CandleDevice;
use burn::backend::{Autodiff, Candle};
use burn::prelude::*;
use llm_from_scratch::model::GptConfig124M;
use tiktoken_rs::r50k_base;

use llm_from_scratch::model::GPTModel;
use llm_from_scratch::tokenizer::{text_to_token_ids, token_ids_to_text};
use llm_from_scratch::train::{TrainConfig, train};

fn main() {
    type Backend = Autodiff<Candle>;
    let device = CandleDevice::Cpu;

    // sandbox::<Backend>(&device);

    let train_config = TrainConfig::default();
    train::<Backend>(&train_config, &device);
}

fn _sandbox<B: Backend>(device: &B::Device) {
    let tokenizer = r50k_base().unwrap();

    let config = GptConfig124M::default().with_context_len(3);
    let model: GPTModel<B> = config.init(device);

    let text = String::from("Hello I am ");
    let token_ids: Tensor<B, 2, Int> = text_to_token_ids(&text, &tokenizer, device).unsqueeze();

    let idx = generate_text_simple(&model, token_ids, 10, config.context_length);

    let decoded = token_ids_to_text(idx.squeeze(0), &tokenizer);
    print!("{decoded}");
}

fn generate_text_simple<B: Backend>(
    model: &GPTModel<B>,
    mut token_ids: Tensor<B, 2, Int>,
    max_new_tokens: usize,
    context_size: usize,
) -> Tensor<B, 2, Int> {
    for _ in 0..max_new_tokens {
        let last_idx = token_ids.dims()[1];
        let first_idx = last_idx.saturating_sub(context_size);

        token_ids = token_ids.slice_dim(1, first_idx..last_idx);

        let logits = model.forward(token_ids.clone()); // size (Batch, Ctx, Vocab)
        let last_idx = logits.dims()[1] - 1;
        let device = &logits.device();
        let last_logits: Tensor<B, 3> = logits.select(1, Tensor::from_ints([last_idx], device));

        let probas: Tensor<B, 2> = burn::tensor::activation::softmax(last_logits.squeeze(0), 1);
        let idx_next: Tensor<B, 1, Int> = probas.argmax(1).squeeze(0);

        token_ids = Tensor::cat(vec![token_ids, idx_next.unsqueeze()], 1);
    }
    token_ids
}
