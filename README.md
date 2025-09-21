# Rust GPT-2

GPT 2 written in Rust using [Burn](https://github.com/tracel-ai/burn).

# Examples

## Train

```bash
cargo run --example train --release
```


# (WIP) Load OpenAI weights

1. Download the weights:
```bash
python3 scripts/gpt_download.py
```
This also converts the tensorflow weights into a `safetensors` file.

*Note: You will need `tensorflow`, `safetensors` and `tqdm` installed
