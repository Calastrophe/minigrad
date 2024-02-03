pub mod engine;
use engine::{MathOps, Value};
use rand::{thread_rng, Rng};

#[derive(Debug)]
pub struct Neuron<T: MathOps<T>> {
    weights: Vec<Value<T>>,
    bias: Value<T>,
    non_linear: bool,
}

impl<T: MathOps<T>> Neuron<T> {
    pub fn new(inputs: usize, non_linear: bool) -> Self {
        let weights = (0..inputs)
            .map(|_| thread_rng().gen_range(-T::one()..T::one()))
            .map(|v| Value::new(v))
            .collect();

        Self {
            weights,
            bias: Value::new(T::zero()),
            non_linear,
        }
    }

    pub fn zero_grad(&mut self) {
        self.weights
            .iter_mut()
            .chain(std::iter::once(&mut self.bias))
            .for_each(|v| v.grad = T::zero())
    }
}

#[derive(Debug)]
pub struct Layer<T: MathOps<T>> {
    neurons: Vec<Neuron<T>>,
}

impl<T: MathOps<T>> Layer<T> {
    pub fn new(inputs: usize, outputs: usize, non_linear: bool) -> Self {
        let neurons = (0..outputs)
            .map(|_| Neuron::new(inputs, non_linear))
            .collect();

        Layer { neurons }
    }

    pub fn zero_grad(&mut self) {
        self.neurons.iter_mut().for_each(|n| n.zero_grad());
    }
}

#[derive(Debug)]
pub struct MLP<T: MathOps<T>> {
    layers: Vec<Layer<T>>,
}

impl<T: MathOps<T>> MLP<T> {
    pub fn new(layers: &[usize]) -> Self {
        let before_last = layers.len() - 2;
        let layers = layers
            .windows(2)
            .enumerate()
            .map(|(index, pair)| Layer::new(pair[0], pair[1], before_last != index))
            .collect();

        MLP { layers }
    }

    pub fn zero_grad(&mut self) {
        self.layers.iter_mut().for_each(|l| l.zero_grad());
    }
}
