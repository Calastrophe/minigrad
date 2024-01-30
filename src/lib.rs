pub mod engine;
use engine::Value;
use num_traits::NumAssign;

pub trait Module<T>
where
    T: NumAssign + Copy,
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
    T: NumAssign + Copy,
{
    weights: Vec<Value<T>>,
    biases: Value<T>,
    non_linear: bool,
}

pub struct Layer<T>
where
    T: NumAssign + Copy,
{
    neurons: Vec<Neuron<T>>,
}

pub struct MLP<T>
where
    T: NumAssign + Copy,
{
    layers: Vec<Layer<T>>,
}
