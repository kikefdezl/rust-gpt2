use std::collections::HashMap;

use burn::prelude::*;
use regex::Regex;
use tiktoken_rs::CoreBPE;

pub const END_OF_TEXT_TOKEN: &str = "<|endoftext|>";
const UNKNOWN_TOKEN: &str = "<|unk|>";

pub fn tokenize(text: &str) -> Vec<&str> {
    let re = Regex::new(r#"([,.:;?_!"()']|--|\s)"#).unwrap();

    let mut result = Vec::new();
    let mut last_end = 0;

    for mat in re.find_iter(text) {
        if mat.start() > last_end {
            let segment = &text[last_end..mat.start()];
            if !segment.trim().is_empty() {
                result.push(segment);
            }
        }
        let delimiter = &text[mat.start()..mat.end()];
        if !delimiter.trim().is_empty() {
            result.push(delimiter);
        }
        last_end = mat.end();
    }

    if last_end < text.len() {
        let segment = &text[last_end..];
        if !segment.trim().is_empty() {
            result.push(segment);
        }
    }
    result
}

pub fn build_vocabulary(mut tokens: Vec<&str>) -> HashMap<&str, usize> {
    tokens.sort();
    tokens.dedup();
    let mut vocabulary: HashMap<&str, usize> = HashMap::new();
    for (i, token) in tokens.into_iter().enumerate() {
        vocabulary.insert(token, i);
    }
    vocabulary.insert(END_OF_TEXT_TOKEN, vocabulary.len());
    vocabulary.insert(UNKNOWN_TOKEN, vocabulary.len());
    vocabulary
}

pub struct SimpleTokenizer {
    str_to_int: HashMap<String, usize>,
    int_to_str: HashMap<usize, String>,
}

impl SimpleTokenizer {
    pub fn from_vocabulary(vocab: &HashMap<&str, usize>) -> Self {
        let mut str_to_int: HashMap<String, usize> = HashMap::new();
        let mut int_to_str: HashMap<usize, String> = HashMap::new();
        for (s, i) in vocab.iter() {
            str_to_int.insert(s.to_string(), *i);
            int_to_str.insert(*i, s.to_string());
        }
        SimpleTokenizer {
            str_to_int,
            int_to_str,
        }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let tokens = tokenize(text);

        let mut ids: Vec<usize> = Vec::new();
        for token in tokens {
            match self.str_to_int.get(token) {
                Some(token_id) => ids.push(*token_id),
                None => ids.push(*self.str_to_int.get(UNKNOWN_TOKEN).unwrap()),
            }
        }
        ids
    }

    pub fn decode(&self, token_ids: &Vec<usize>) -> String {
        let mut text = String::new();
        for id in token_ids {
            text.push_str(self.int_to_str.get(id).expect("token id not in vocabulary"));
            text.push(' ');
        }
        let re = Regex::new(r#"\s+([,.?!"()'])"#).unwrap();
        re.replace_all(&text, "$1").to_string()
    }
}

pub fn text_to_token_ids<B: Backend>(
    text: &str,
    tokenizer: &CoreBPE,
    device: &B::Device,
) -> Tensor<B, 1, Int> {
    let token_ids = tokenizer.encode_ordinary(text);
    let length = token_ids.len();
    let tensor_data = TensorData::new(token_ids, vec![length]);
    Tensor::from_data(tensor_data, device)
}

pub fn token_ids_to_text<B: Backend>(token_ids: Tensor<B, 1, Int>, tokenizer: &CoreBPE) -> String {
    let ids: Vec<i64> = token_ids.into_data().into_vec().unwrap_or_default();
    let ids_casted: Vec<u32> = ids.into_iter().map(|x| x as u32).collect();
    // tokenizer.decode(ids_casted).unwrap_or(String::from(""))
    tokenizer.decode(ids_casted).unwrap()
}
