use llm_from_scratch::config;
use llm_from_scratch::tokenizer::{END_OF_TEXT_TOKEN, SimpleTokenizer, build_vocabulary, tokenize};

use std::fs::read_to_string;

fn main() {
    let text = read_to_string(config::RAW_DATA_FILE).unwrap();
    let tokens = tokenize(&text);
    let vocabulary = build_vocabulary(tokens);
    let tokenizer = SimpleTokenizer::from_vocabulary(&vocabulary);
    let text1 = "Hello, do you like tea?";
    let text2 = "In the sunlit terraces of the palace.";
    let text = [text1, text2].join(&format!(" {END_OF_TEXT_TOKEN} "));
    let ids = tokenizer.encode(&text);

    let text = tokenizer.decode(&ids);
    println!("{text}");
}
