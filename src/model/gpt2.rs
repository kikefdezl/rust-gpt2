use std::f32::consts::PI;

use super::attention::MultiHeadAttention;
use burn::config::Config;
use burn::module::{Module, Param};
use burn::nn::{self, Dropout};
use burn::prelude::*;
use burn::tensor::backend::Backend;

#[derive(Config)]
pub struct Gpt2Config {
    #[config(default = 50257)]
    pub vocab_size: usize,
    #[config(default = 1024)]
    pub context_length: usize,
    #[config(default = 768)]
    pub embedding_dim: usize,
    #[config(default = 12)]
    pub n_heads: usize,
    #[config(default = 12)]
    pub n_layers: usize,
    #[config(default = 0.1)]
    pub embedding_drop_rate: f64,
    #[config(default = 0.1)]
    pub attention_drop_rate: f64,
    #[config(default = 0.1)]
    pub shortcut_layer_drop_rate: f64,
    #[config(default = false)]
    pub qkv_bias: bool,
}

impl Gpt2Config {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Gpt2<B> {
        let token_embedding: nn::Embedding<B> =
            nn::EmbeddingConfig::new(self.vocab_size, self.embedding_dim).init(device);
        let positional_embedding: nn::Embedding<B> =
            nn::EmbeddingConfig::new(self.context_length, self.embedding_dim).init(device);

        let dropout_embedding = nn::DropoutConfig {
            prob: self.embedding_drop_rate,
        }
        .init();

        let transformer_blocks: Vec<TransformerBlock<B>> = (0..self.n_layers)
            .map(|_| {
                TransformerBlock::new(
                    self.embedding_dim,
                    self.n_heads,
                    self.attention_drop_rate,
                    self.shortcut_layer_drop_rate,
                    device,
                )
            })
            .collect();

        let norm = LayerNorm::new(self.embedding_dim, device);

        let out = nn::LinearConfig::new(self.embedding_dim, self.vocab_size)
            .with_bias(false)
            .init(device);

        Gpt2 {
            token_embedding,
            positional_embedding,
            dropout_embedding,
            transformer_blocks,
            norm,
            linear_out: out,
        }
    }
}

#[derive(Module, Debug)]
pub struct Gpt2<B: Backend> {
    token_embedding: nn::Embedding<B>,
    positional_embedding: nn::Embedding<B>,
    dropout_embedding: nn::Dropout,
    transformer_blocks: Vec<TransformerBlock<B>>,
    norm: LayerNorm<B>,
    linear_out: nn::Linear<B>,
}

impl<B: Backend> Gpt2<B> {
    /// in_idx is a tensor of shape [BatchSize, ContextLen]
    pub fn forward(&self, in_idx: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch_size, seq_len] = in_idx.dims();
        let device = &in_idx.device();

        let token_embeddings = self.token_embedding.forward(in_idx);

        let positional_embeddings = self.positional_embedding.forward(
            Tensor::arange(0..seq_len as i64, device)
                .unsqueeze()
                .repeat(&[batch_size, 0]),
        );

        let x = token_embeddings + positional_embeddings;

        let mut x = self.dropout_embedding.forward(x);
        for block in &self.transformer_blocks {
            x = block.forward(x);
        }

        let x = self.norm.forward(x);

        self.linear_out.forward(x)
    }
}

#[derive(Module, Debug)]
struct TransformerBlock<B: Backend> {
    norm1: LayerNorm<B>,
    mha: MultiHeadAttention<B>,
    norm2: LayerNorm<B>,
    ff: FeedForward<B>,
    dropout: Dropout,
}

impl<B: Backend> TransformerBlock<B> {
    fn new(
        embedding_dim: usize,
        num_heads: usize,
        attention_drop_rate: f64,
        shortcut_drop_rate: f64,
        device: &B::Device,
    ) -> TransformerBlock<B> {
        let norm1 = LayerNorm::new(embedding_dim, device);
        let mha = MultiHeadAttention::new(
            embedding_dim,
            embedding_dim,
            num_heads,
            attention_drop_rate,
            false,
            device,
        );
        let dropout = nn::DropoutConfig {
            prob: shortcut_drop_rate,
        }
        .init();
        let norm2 = LayerNorm::new(embedding_dim, device);
        let ff = FeedForward::new(embedding_dim, device);
        TransformerBlock {
            norm1,
            mha,
            dropout,
            norm2,
            ff,
        }
    }
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let shortcut = x.clone();
        let x = self.norm1.forward(x);
        let x = self.mha.forward(x);
        let x = self.dropout.forward(x);
        let x = x + shortcut;

        let shortcut = x.clone();
        let x = self.norm2.forward(x);
        let x = self.ff.forward(x);
        let x = self.dropout.forward(x);
        x + shortcut
    }
}

#[derive(Module, Debug)]
struct LayerNorm<B: Backend> {
    eps: f32,
    scale: Param<Tensor<B, 3>>,
    shift: Param<Tensor<B, 3>>,
}

impl<B: Backend> LayerNorm<B> {
    fn new(embedding_dim: usize, device: &B::Device) -> LayerNorm<B> {
        let eps = 1e-5;
        let scale = Param::from_tensor(Tensor::ones([1, 1, embedding_dim], device));
        let shift = Param::from_tensor(Tensor::zeros([1, 1, embedding_dim], device));
        LayerNorm { eps, scale, shift }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let (var, mean) = x.clone().var_mean(2);
        let norm_x: Tensor<B, 3> = (x - mean) / (var + self.eps).sqrt();
        self.scale.val().mul(norm_x).add(self.shift.val())
    }
}

#[derive(Module, Debug)]
struct FeedForward<B: Backend> {
    pre: nn::Linear<B>,
    post: nn::Linear<B>,
}

impl<B: Backend> FeedForward<B> {
    fn new(embedding_dim: usize, device: &B::Device) -> FeedForward<B> {
        FeedForward {
            pre: nn::LinearConfig::new(embedding_dim, 4 * embedding_dim).init(device),
            post: nn::LinearConfig::new(4 * embedding_dim, embedding_dim).init(device),
        }
    }
    fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let x = self.pre.forward(x);
        let x = gelu(x);
        self.post.forward(x)
    }
}

/// 0.5 * x * (1 + tanh(sqrt(2.0 / pi) * (x + 0.044715 * x^3))))
fn gelu<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    x.clone().mul_scalar(0.5).mul(
        (x.clone()
            .add_scalar(0.044715)
            .mul(x.powf_scalar(3.0))
            .mul_scalar((2.0 / PI).sqrt()))
        .tanh()
        .add_scalar(1),
    )
}
