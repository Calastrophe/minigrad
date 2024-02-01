pub mod engine;
use engine::Value;
use num_traits::{NumAssign, Pow};
use std::ops::Neg;

pub trait Module<T>
where
    T: NumAssign + Copy + PartialOrd + Pow<T, Output = T> + Neg<Output = T>,
{
    fn zero_grad(&mut self) {
        for mut p in self.parameters() {
            p.grad = T::zero();
        }
    }

    fn parameters(&mut self) -> impl Iterator<Item = Value<T>>;
}

pub struct Neuron<T>
where
    T: NumAssign + Copy + PartialOrd + Pow<T, Output = T> + Neg<Output = T>,
{
    weights: Vec<Value<T>>,
    biases: Value<T>,
    non_linear: bool,
}

pub struct Layer<T>
where
    T: NumAssign + Copy + PartialOrd + Pow<T, Output = T> + Neg<Output = T>,
{
    neurons: Vec<Neuron<T>>,
}

pub struct MLP<T>
where
    T: NumAssign + Copy + PartialOrd + Pow<T, Output = T> + Neg<Output = T>,
{
    layers: Vec<Layer<T>>,
}
