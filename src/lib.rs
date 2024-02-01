pub mod engine;
use engine::{MathOps, Value};

pub trait Module<T: MathOps<T>> {
    fn zero_grad(&mut self) {
        for mut p in self.parameters() {
            p.grad = T::zero();
        }
    }

    fn parameters(&mut self) -> impl Iterator<Item = Value<T>>;
}

pub struct Neuron<T: MathOps<T>> {
    weights: Vec<Value<T>>,
    biases: Value<T>,
    non_linear: bool,
}

pub struct Layer<T: MathOps<T>> {
    neurons: Vec<Neuron<T>>,
}

pub struct MLP<T: MathOps<T>> {
    layers: Vec<Layer<T>>,
}
