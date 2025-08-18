use burn::module::Module;
use burn::nn;
use burn::prelude::*;
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
        let attn_scaled = softmax(attn / d_k.sqrt(), 1);
        attn_scaled.matmul(v)
    }
}

#[derive(Module, Debug)]
pub struct CausalSelfAttention<B: Backend> {
    w_query: nn::Linear<B>,
    w_key: nn::Linear<B>,
    w_value: nn::Linear<B>,
    d_out: usize,
}

impl<B: Backend> CausalSelfAttention<B> {
    pub fn new(d_in: usize, d_out: usize, qkv_bias: bool, device: &B::Device) -> Self {
        let query_config = nn::LinearConfig::new(d_in, d_out).with_bias(qkv_bias);
        let key_config = nn::LinearConfig::new(d_in, d_out).with_bias(qkv_bias);
        let value_config = nn::LinearConfig::new(d_in, d_out).with_bias(qkv_bias);

        Self {
            w_query: query_config.init(device),
            w_key: key_config.init(device),
            w_value: value_config.init(device),
            d_out,
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, t, d_in] = input.dims();

        let q = self.w_query.forward(input.clone());
        let k = self.w_key.forward(input.clone());
        let v = self.w_value.forward(input);

        let attn = q.matmul(k.transpose());

        // the tril_mask function doesn't respect the batch size so we have
        // to repeat manually. It also doesn't broadcast properly in mask_fill()
        let mask: Tensor<B, 3, Bool> = Tensor::tril_mask([b, t, t], 0, &attn.device());
        let mask = mask.repeat(&[b, 0]);

        let attn_masked = attn.mask_fill(mask, f32::NEG_INFINITY);

        let attn_scaled = softmax(attn_masked / (d_in as f64).sqrt(), 1);

        attn_scaled.matmul(v)
    }
}

#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    d_out: usize,
    n_heads: usize,
    head_dim: usize,
    w_query: nn::Linear<B>,
    w_key: nn::Linear<B>,
    w_value: nn::Linear<B>,
    dropout: nn::Dropout,
    out_proj: nn::Linear<B>,
}

impl<B: Backend> MultiHeadAttention<B> {
    pub fn new(
        d_in: usize,
        d_out: usize,
        num_heads: usize,
        qkv_bias: bool,
        device: &B::Device,
    ) -> Self {
        assert!(
            d_out % num_heads == 0,
            "num_heads must be divisible by d_out"
        );
        let query_config = nn::LinearConfig::new(d_in, d_out).with_bias(qkv_bias);
        let key_config = nn::LinearConfig::new(d_in, d_out).with_bias(qkv_bias);
        let value_config = nn::LinearConfig::new(d_in, d_out).with_bias(qkv_bias);

        let dropout = nn::DropoutConfig { prob: 0.2 }.init();

        let out_proj = nn::LinearConfig::new(d_in, d_out).init(device);

        Self {
            d_out,
            n_heads: num_heads,
            head_dim: d_out / num_heads,
            w_query: query_config.init(device),
            w_key: key_config.init(device),
            w_value: value_config.init(device),
            dropout,
            out_proj,
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, t, d_in] = input.dims();

        let q = self.w_query.forward(input.clone());
        let k = self.w_key.forward(input.clone());
        let v = self.w_value.forward(input);

        let q = q
            .reshape([b, t, self.n_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([b, t, self.n_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([b, t, self.n_heads, self.head_dim])
            .swap_dims(1, 2);

        let attn = q.matmul(k.transpose()); // [b, num_heads, t, t]

        // tril_mask always produces shapes [1, ..., N, N] regardless of additional dims we pass,
        // so we broadcast manually for dims 0 and 1.
        let mask: Tensor<B, 2, Bool> = Tensor::tril_mask([t, t], 0, &attn.device());
        let mask = mask.unsqueeze::<3>().unsqueeze::<4>().repeat(&[b, self.n_heads, 0, 0]);
        let attn_masked = attn.mask_fill(mask, f64::NEG_INFINITY);

        let attn_scaled = softmax(attn_masked / (d_in as f64).sqrt(), 3);
        let attn_scaled = self.dropout.forward(attn_scaled);

        let context_vec = attn_scaled.matmul(v).swap_dims(1, 2);
        let context_vec = context_vec.reshape([b, t, self.d_out]);
        self.out_proj.forward(context_vec)
    }
}
