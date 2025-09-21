use burn::backend::candle::{Candle, CandleDevice};
use burn::prelude::*;
use rust_gpt2::utils::multinomial_single;
use std::path::PathBuf;
use tiktoken_rs::r50k_base;

use rust_gpt2::load::load_gpt2;
use rust_gpt2::model::gpt2::Gpt2;
use rust_gpt2::tokenization::{text_to_token_ids, token_ids_to_text};

const WEIGHTS_FILE: &str = "../gpt2/gpt2.safetensors";

fn main() {
    type Backend = Candle;
    let device = CandleDevice::Cpu;

    let tokenizer = r50k_base().unwrap();
    let gpt2 = load_gpt2(PathBuf::from(WEIGHTS_FILE), &device);

    let text = String::from("Every effort moves you ");
    let token_ids: Tensor<Backend, 2, Int> =
        text_to_token_ids(&text, &tokenizer, &device).unsqueeze();
    let idx = generate(&gpt2, token_ids, 25, 256, 0.8, 5);
    let decoded = token_ids_to_text(idx.squeeze(0), &tokenizer);

    print!("{decoded}");
}

fn generate<B: Backend>(
    model: &Gpt2<B>,
    mut token_ids: Tensor<B, 2, Int>,
    max_new_tokens: usize,
    context_size: usize,
    temperature: f64,
    top_k: usize,
) -> Tensor<B, 2, Int> {
    assert!(temperature >= 0.0);
    assert!(top_k >= 1);
    for _ in 0..max_new_tokens {
        let last_idx = token_ids.dims()[1];
        let first_idx = last_idx.saturating_sub(context_size);

        token_ids = token_ids.slice_dim(1, first_idx..last_idx);

        let logits = model.forward(token_ids.clone()); // size (Batch, Ctx, Vocab)
        let last_idx = logits.dims()[1] - 1;
        let device = &logits.device();
        let last_logits: Tensor<B, 1> = logits
            .select(1, Tensor::from_ints([last_idx], device))
            .squeeze::<2>(0)
            .squeeze(0);

        // temperature scaling
        let last_logits_scaled = last_logits.div_scalar(temperature);

        // top k sampling
        let top_k_logits = last_logits_scaled.clone().topk(top_k, 0);
        let last_index: usize = top_k_logits.dims()[0] - 1;
        let mask = last_logits_scaled.clone().lower_elem(last_index as u64);
        let last_logits_masked = last_logits_scaled.mask_fill(mask, 1e-8);

        // probability based selection
        let probabilities = burn::tensor::activation::softmax(last_logits_masked, 0);
        let idx_next: Tensor<B, 1, Int> = multinomial_single(probabilities);

        token_ids = Tensor::cat(vec![token_ids, idx_next.unsqueeze()], 1);
    }
    println!("{}", token_ids);
    token_ids
}
