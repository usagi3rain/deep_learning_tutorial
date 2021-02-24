use ndarray::prelude::*;
use ndarray_stats::QuantileExt;

pub fn exp<D: Dimension>(x: &Array<f64, D>) -> Array<f64, D> {
    x.map(|x| x.exp())
}

pub fn sigmoid(x: &Array2<f64>) -> Array2<f64> {
    1.0 / (1.0 + exp(&(-x)))
}

pub fn sigmoid_grad(x: &Array2<f64>) -> Array2<f64> {
    (1.0 - sigmoid(&x)) * sigmoid(&x)
}

pub fn softmax(x: &Array2<f64>) -> Array2<f64> {
    let mut s = Vec::new();

    for i in 0..x.shape()[0] {
        let x = &x.row(i).to_owned();
        let x_max = *x.max().unwrap();
        let exp_a = exp(&(x - x_max));
        let exp_sum_a = exp_a.sum();

        s.append(&mut (exp_a / exp_sum_a).to_vec());
    }

    Array::from_shape_vec(x.raw_dim(), s).unwrap()
}

pub fn log<D: Dimension>(x: &Array<f64, D>) -> Array<f64, D> {
    x.map(|x| x.ln())
}

pub fn cross_entropy_error<D: Dimension>(y: &Array<f64, D>, t: &Array<f64, D>) -> f64 {
    -(t * &(log(&(1e-7 + y)))).sum() / y.shape()[0] as f64
}
