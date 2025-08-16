use burn::module::{Module, Param};
use burn::prelude::*;
use burn::tensor::Distribution;
use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
pub struct SelfAttentionV1<B: Backend> {
    w_query: Param<Tensor<B, 2>>,
    w_key: Param<Tensor<B, 2>>,
    w_value: Param<Tensor<B, 2>>,
}

impl<B: Backend> SelfAttentionV1<B> {
    pub fn new(d_in: usize, d_out: usize, device: &B::Device) -> Self {
        let distribution = Distribution::Uniform(0.0, 1.0);

        let w_query: Tensor<B, 2> = Tensor::random([d_in, d_out], distribution, device);
        let w_key: Tensor<B, 2> = Tensor::random([d_in, d_out], distribution, device);
        let w_value: Tensor<B, 2> = Tensor::random([d_in, d_out], distribution, device);

        let w_query = Param::from_tensor(w_query).set_require_grad(false);
        let w_key = Param::from_tensor(w_key).set_require_grad(false);
        let w_value = Param::from_tensor(w_value).set_require_grad(false);

        Self {
            w_query,
            w_key,
            w_value,
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let q = input.clone().matmul(self.w_query.val());
        let k = input.clone().matmul(self.w_key.val());
        let v = input.matmul(self.w_value.val());

        let attn_scores = q.matmul(k.transpose());
        let attn_scores = burn::tensor::activation::softmax(attn_scores, 1);

        attn_scores.matmul(v)
    }
}
