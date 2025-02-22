
fn empty<T>(shape: Vec<usize>) -> Vec<T> {
    let numel = shape.iter().product();
    let mut t = vec![];
    t.resize(numel, Default::default());
}
