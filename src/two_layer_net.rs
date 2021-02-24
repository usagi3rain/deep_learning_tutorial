use ndarray::{prelude::*, Zip};
use ndarray_stats::QuantileExt;

mod function;

pub struct Perceptron {
    weight: Vec<Array2<f64>>,
    bias: Vec<Array1<f64>>,
}

impl Perceptron {
    pub fn new(size: (usize, usize, usize), weight_init_std: f64) -> Perceptron {
        let mut weight = Vec::new();
        let w1 = Array::from_shape_fn((size.0, size.1).f(), |_| {
            rand::random::<f64>() * weight_init_std
        });
        let w2 = Array::from_shape_fn((size.1, size.2).f(), |_| {
            rand::random::<f64>() * weight_init_std
        });
        weight.push(w1);
        weight.push(w2);

        let mut bias = Vec::new();
        let b1 = Array::zeros(size.1);
        let b2 = Array::zeros(size.2);
        bias.push(b1);
        bias.push(b2);

        Perceptron { weight, bias }
    }

    pub fn print(&self) {
        println!("w1: {}", self.weight[0]);
        println!("w2: {}", self.weight[1]);
        println!("b1: {}", self.bias[0]);
        println!("b2: {}", self.bias[1]);
    }
}

pub struct TwoLayerNet {
    params: Perceptron,
    grads: Perceptron,
}

impl TwoLayerNet {
    pub fn new(size: (usize, usize, usize), weight_init_std: f64) -> TwoLayerNet {
        TwoLayerNet {
            params: Perceptron::new(size, weight_init_std),
            grads: Perceptron::new(size, weight_init_std),
        }
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array2<f64> {
        let a1 = x.dot(&self.params.weight[0]) + &self.params.bias[0];
        let z1 = function::sigmoid(&a1);
        let a2 = z1.dot(&self.params.weight[1]) + &self.params.bias[1];

        function::softmax(&a2)
    }

    pub fn loss(&self, x: &Array2<f64>, t: &Array2<f64>) -> f64 {
        let y = self.predict(x);

        function::cross_entropy_error::<Ix2>(&y, t)
    }

    pub fn accuracy(&self, x: &Array2<f64>, t: &Array2<f64>) -> f64 {
        let y = self.predict(x);
        let y: Array1<usize> = y.outer_iter().map(|x| x.argmax().unwrap()).collect();
        let t: Array1<usize> = t.outer_iter().map(|x| x.argmax().unwrap()).collect();

        let mut new: Array1<usize> = Array::zeros(y.raw_dim());

        Zip::from(&mut new).and(&y).and(&t).apply(|new, &y, &t| {
            if y == t {
                *new += 1
            }
        });

        new.sum() as f64 / x.shape()[0] as f64
    }

    pub fn gradient(&mut self, x: &Array2<f64>, t: &Array2<f64>) {
        let a1 = x.dot(&self.params.weight[0]) + &self.params.bias[0];
        let z1 = function::sigmoid(&a1);
        let a2 = z1.dot(&self.params.weight[1]) + &self.params.bias[1];

        let y = function::softmax(&a2);

        let dy = (y - t) / x.shape()[0] as f64;
        let w2 = z1.t().dot(&dy);
        let b2 = dy.sum_axis(Axis(0));
        let dz1 = dy.dot(&self.params.weight[1].t());
        let da1 = function::sigmoid_grad(&a1) * &dz1;
        let w1 = x.t().dot(&da1);
        let b1 = da1.sum_axis(Axis(0));

        self.grads.weight[0] = w1;
        self.grads.weight[1] = w2;
        self.grads.bias[0] = b1;
        self.grads.bias[1] = b2;
    }

    pub fn update_params(&mut self, learning_rate: f64) {
        self.params.weight[0] -= &(learning_rate * &self.grads.weight[0]);
        self.params.weight[1] -= &(learning_rate * &self.grads.weight[1]);
        self.params.bias[0] -= &(learning_rate * &self.grads.bias[0]);
        self.params.bias[1] -= &(learning_rate * &self.grads.bias[1]);
    }

    pub fn print(&self) {
        println!("params:");
        self.params.print();
        println!("grads:");
        self.grads.print();
    }
}
