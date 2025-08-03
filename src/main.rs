use llm_from_scratch::config;
use llm_from_scratch::tokenizer::{build_vocabulary, tokenize};

use std::fs::read_to_string;

fn main() {
    let text = read_to_string(config::RAW_DATA_FILE).unwrap();
    let tokens = tokenize(&text);
    println!("Tokenized text into {} tokens", tokens.len());
    let vocabulary = build_vocabulary(tokens);
    // for (token, i) in vocabulary.iter() {
    //     if *i < 50 {
    //         println!("{i} {token}")
    //     }
    // }
}
