use llm_from_scratch::config;
use llm_from_scratch::tokenizer::tokenize;

use std::fs::read_to_string;

fn main() {
    let text = read_to_string(config::RAW_DATA_FILE).unwrap();
    let tokens = tokenize(&text);
    println!("Tokenized text into {} tokens", tokens.len());
}
