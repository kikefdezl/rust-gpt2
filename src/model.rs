use burn::module::Module;
use burn::nn;
use burn::prelude::*;
use burn::tensor::backend::Backend;

pub struct GptConfig124M {
    pub vocab_size: usize,
    pub context_length: usize,
    pub embedding_dim: usize,
    pub n_heads: u32,
    pub n_layers: u32,
    pub drop_rate: f32,
    pub qkv_bias: bool,
}

impl Default for GptConfig124M {
    fn default() -> Self {
        Self {
            vocab_size: 50257,
            context_length: 1024,
            embedding_dim: 768,
            n_heads: 12,
            n_layers: 12,
            drop_rate: 0.1,
            qkv_bias: false,
        }
    }
}

impl GptConfig124M {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DummyGPTModel<B> {
        let token_embedding: nn::Embedding<B> =
            nn::EmbeddingConfig::new(self.vocab_size, self.embedding_dim).init(device);
        let positional_embedding: nn::Embedding<B> =
            nn::EmbeddingConfig::new(self.context_length, self.embedding_dim).init(device);

        let dropout_embedding = nn::DropoutConfig { prob: 0.2 }.init();

        let transformer_blocks: Vec<DummyTransformerBlock> =
            (0..12).map(|_| DummyTransformerBlock::new()).collect();

        let norm = DummyLayerNorm::new();

        let out = nn::LinearConfig::new(self.embedding_dim, self.vocab_size)
            .with_bias(false)
            .init(device);

        DummyGPTModel {
            token_embedding,
            positional_embedding,
            dropout_embedding,
            transformer_blocks,
            norm,
            out,
        }
    }
}

#[derive(Module, Debug)]
pub struct DummyGPTModel<B: Backend> {
    token_embedding: nn::Embedding<B>,
    positional_embedding: nn::Embedding<B>,
    dropout_embedding: nn::Dropout,
    transformer_blocks: Vec<DummyTransformerBlock>,
    norm: DummyLayerNorm,
    out: nn::Linear<B>,
}

impl<B: Backend> DummyGPTModel<B> {
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

        self.out.forward(x)
    }
}

#[derive(Module, Debug, Clone)]
struct DummyTransformerBlock {}

impl DummyTransformerBlock {
    fn new() -> DummyTransformerBlock {
        DummyTransformerBlock {}
    }
    fn forward<B: Backend>(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        x
    }
}

#[derive(Module, Debug, Clone)]
struct DummyLayerNorm {}

impl DummyLayerNorm {
    fn new() -> DummyLayerNorm {
        DummyLayerNorm {}
    }
    fn forward<B: Backend>(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        x
    }
}
