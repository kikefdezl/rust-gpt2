use bpe_openai::cl100k_base;
use burn::data::dataset::Dataset;
use llm_from_scratch::config;
use llm_from_scratch::data::GPTDatasetV1;

use std::fs::read_to_string;

fn main() {
    let text = read_to_string(config::RAW_DATA_FILE).unwrap();

    let tokenizer = cl100k_base();
    let ids = tokenizer.encode(&text);
    let ids: Vec<u32> = ids.into_iter().skip(50).collect();

    let dataset = GPTDatasetV1::new(&ids, 256, 128);
    for i in 0..dataset.len() {
        println!("{:?}", dataset.get(i).unwrap());
    }
}
