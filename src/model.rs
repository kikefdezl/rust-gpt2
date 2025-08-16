use burn::module::Module;
use burn::nn;
use burn::prelude::*;
use burn::tensor::Distribution;
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
pub struct SelfAttention<B: Backend> {
    w_query: nn::Linear<B>,
    w_key: nn::Linear<B>,
    w_value: nn::Linear<B>,
}

impl<B: Backend> SelfAttention<B> {
    pub fn new(d_in: usize, d_out: usize, qkv_bias: bool, device: &B::Device) -> Self {
        let query_config = nn::LinearConfig::new(d_in, d_out).with_bias(qkv_bias);
        let key_config = nn::LinearConfig::new(d_in, d_out).with_bias(qkv_bias);
        let value_config = nn::LinearConfig::new(d_in, d_out).with_bias(qkv_bias);

        Self {
            w_query: query_config.init(device),
            w_key: key_config.init(device),
            w_value: value_config.init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let q = input.clone().matmul(self.w_query.weight.val());
        let k = input.clone().matmul(self.w_key.weight.val());
        let v = input.matmul(self.w_value.weight.val());

        let d_k = k.shape().dims[1] as f64;

        let attn = q.matmul(k.transpose());
        let attn_scaled = softmax(attn.div_scalar(d_k.sqrt()), 1);
        attn_scaled.matmul(v)
    }
}
