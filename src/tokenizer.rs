use regex::Regex;

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
