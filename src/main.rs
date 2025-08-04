use llm_from_scratch::config;
use llm_from_scratch::tokenizer::{SimpleTokenizerV1, build_vocabulary, tokenize};

use std::fs::read_to_string;

fn main() {
    let text = read_to_string(config::RAW_DATA_FILE).unwrap();
    let tokens = tokenize(&text);
    let vocabulary = build_vocabulary(tokens);
    let tokenizer = SimpleTokenizerV1::from_vocabulary(&vocabulary);
    let text = "Hello, do you like tea?";
    let ids = tokenizer.encode(text);

    let text = tokenizer.decode(&ids);
    println!("{text}");
}
