use crate::Value;
use rand::{thread_rng, Rng};

#[derive(Debug)]
pub struct Neuron {
    weights: Vec<Value>,
    bias: Value,
    non_linear: bool,
}

impl Neuron {
    pub fn new(inputs: usize, non_linear: bool) -> Self {
        let weights = (0..inputs)
            .map(|_| thread_rng().gen_range(-1.0..1.0))
            .map(|v| Value::new(v))
            .collect();

        Self {
            weights,
            bias: Value::new(0.0),
            non_linear,
        }
    }

    pub fn zero_grad(&mut self) {
        self.weights
            .iter_mut()
            .chain(std::iter::once(&mut self.bias))
            .for_each(|v| v.grad = 0.0)
    }
}
