use crate::Neuron;

#[derive(Debug)]
pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
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
