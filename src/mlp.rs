use crate::Layer;

#[derive(Debug)]
pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
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
