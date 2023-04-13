use rand::distributions::{Distribution, Uniform};

use crate::Value;
use std::iter;

pub struct Neuron {
    weights: Vec<Value>,
    bias: Value,
    nonlinearity: bool,
}

impl Neuron {
    pub fn new(in_channels: usize, nonlinearity: bool) -> Self {
        let mut rng = rand::thread_rng();
        let uniform = Uniform::from(-1.0..1.0);
        Self {
            weights: iter::repeat_with(|| Value::leaf(uniform.sample(&mut rng)))
                .take(in_channels)
                .collect(),
            bias: Value::leaf(uniform.sample(&mut rng)),
            nonlinearity,
        }
    }

    pub fn forward(&self, inputs: &[Value]) -> Value {
        let preact = self
            .weights
            .iter()
            .zip(inputs)
            .fold(self.bias.clone(), |preact, (weight, input)| {
                preact + weight * input
            });
        if self.nonlinearity {
            preact.relu()
        } else {
            preact
        }
    }

    pub fn parameters(&self) -> impl Iterator<Item = &Value> {
        self.weights.iter().chain(iter::once(&self.bias))
    }
}

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(in_channels: usize, out_channels: usize, nonlinearity: bool) -> Self {
        Self {
            neurons: iter::repeat_with(|| Neuron::new(in_channels, nonlinearity))
                .take(out_channels)
                .collect(),
        }
    }

    pub fn forward(&self, inputs: &[Value]) -> Vec<Value> {
        self.neurons
            .iter()
            .map(|neuron| neuron.forward(inputs))
            .collect()
    }

    pub fn parameters(&self) -> impl Iterator<Item = &Value> {
        self.neurons.iter().flat_map(|neuron| neuron.parameters())
    }
}

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(in_channels: usize, out_channels: &[usize]) -> Self {
        let mut layers = Vec::new();
        let mut in_channels = in_channels;
        for (i, &out_chans) in out_channels.iter().enumerate() {
            layers.push(Layer::new(
                in_channels,
                out_chans,
                // Don't apply a nonlinearity to the last layer
                i != out_channels.len() - 1,
            ));
            in_channels = out_chans;
        }
        Self { layers }
    }

    pub fn forward(&self, inputs: &[Value]) -> Vec<Value> {
        self.layers
            .iter()
            .fold(inputs.to_vec(), |inputs, layer| layer.forward(&inputs))
    }

    pub fn parameters(&self) -> impl Iterator<Item = &Value> {
        self.layers.iter().flat_map(|layer| layer.parameters())
    }

    pub fn zero_grad(&self) {
        self.parameters().for_each(|param| param.zero_grad());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let xs = [
            [2.0, 3.0, -1.0],
            [3.0, -1.0, 0.5],
            [0.5, 1.0, 1.0],
            [1.0, 1.0, -1.0],
        ]
        .map(|x| x.map(Value::leaf));
        let ys = [1.0, -1.0, -1.0, 1.0].map(Value::leaf);

        let mlp = MLP::new(3, &[4, 4, 1]);

        for epoch in 0..16 {
            mlp.zero_grad();

            // Compute mean squared error
            let loss = xs
                .iter()
                .map(|x| mlp.forward(x)[0].clone())
                .zip(ys.iter())
                .fold(Value::leaf(0.0), |loss, (y_pred, y_actual)| {
                    loss + (y_pred - y_actual.clone()).pow(2.0)
                });
            println!("Epoch {:0>2}: loss = {:.4}", epoch, loss.data());

            loss.backward();

            mlp.parameters()
                .for_each(|param| param.set_data(param.data() - 0.01 * param.grad()))
        }
    }
}
