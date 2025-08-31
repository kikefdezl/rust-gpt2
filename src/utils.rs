use burn::prelude::*;
use rand::Rng;

/// Given a rank 1 tensor of probabilities, randomly returns the index of an
/// element following the probability distribution.
///
/// E.g. for tensor [0.9, 0.05, 0.05], this function will return `0` 90% of the time,
/// `1` 5% of the time, and `2` 5% of the time.
pub fn multinomial_single<B: Backend>(probs: Tensor<B, 1>) -> Tensor<B, 1, Int> {
    let device = &probs.device();

    // normalization
    let probs = probs.clone() / probs.sum();

    // Host copy
    let values: Vec<f32> = probs.into_data().to_vec().unwrap();
    let sum: f32 = values.iter().sum();
    let normed: Vec<f32> = values.iter().map(|&x| x / sum).collect();

    // Build CDF
    let mut acc = 0.0;
    let mut cdf = Vec::with_capacity(normed.len());
    for p in normed {
        acc += p;
        cdf.push(acc);
    }

    let u: f32 = rand::rng().random();
    let idx = cdf.iter().position(|&x| x >= u).unwrap_or(cdf.len() - 1);
    Tensor::<B, 1, Int>::from_ints([idx as i32], device)
}
