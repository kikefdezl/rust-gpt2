use bpe_openai::cl100k_base;
use llm_from_scratch::config;
use llm_from_scratch::tokenizer::{END_OF_TEXT_TOKEN, build_vocabulary, tokenize};

use std::fs::read_to_string;

fn main() {
    let text = read_to_string(config::RAW_DATA_FILE).unwrap();
    let tokens = tokenize(&text);
    let vocabulary = build_vocabulary(tokens);

    let tokenizer = cl100k_base();
    let text1 = "Hello, do you like tea?";
    let text2 = "In the sunlit terraces of the palace.";
    let text = [text1, text2].join(&format!(" {END_OF_TEXT_TOKEN} "));
    let ids = tokenizer.encode(&text);

    println!("{ids:?}");
    let text = tokenizer.decode(&ids).unwrap();
    println!("{text}");
}
