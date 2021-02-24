use mnist::{MnistBuilder, NormalizedMnist};
use ndarray::prelude::*;
use ndarray::Array;

mod data;
mod twoLayerNet;

const TRN_SIZE: u32 = 50;
const ROWS: u32 = 28;
const COLS: u32 = 28;

fn init_network() -> (Vec<Array2<f32>>, Vec<Array1<f32>>) {
    let mut weight = Vec::new();
    let mut bias = Vec::new();

    weight.push(Array::from_shape_vec(((ROWS * COLS) as usize, 50).f(), data::w1()).unwrap());
    weight.push(Array::from_shape_vec((50, 100).f(), data::w2()).unwrap());
    weight.push(Array::from_shape_vec((100, 10).f(), data::w3()).unwrap());
    bias.push(ndarray::arr1(&data::b1()));
    bias.push(ndarray::arr1(&data::b2()));
    bias.push(ndarray::arr1(&data::b3()));

    (weight, bias)
}

fn forward(network: &(Vec<Array2<f32>>, Vec<Array1<f32>>), x: &Array1<f32>) -> Array1<f32> {
    let (weight, bias) = network;

    let a1 = x.dot(&weight[0]) + &bias[0];
    let z1 = sigmoid(a1);
    let a2 = z1.dot(&weight[1]) + &bias[1];
    let z2 = sigmoid(a2);
    let a3 = z2.dot(&weight[2]) + &bias[2];

    softmax(a3)
}

fn exp(x: &Array1<f32>) -> Array1<f32> {
    x.map(|x| x.exp())
}

fn sigmoid(x: Array1<f32>) -> Array1<f32> {
    let ones = Array::ones(x.raw_dim());
    let minus = Array::zeros(x.raw_dim()) - x;
    ones.clone() / (ones + exp(&minus))
}

fn identity_function(x: Array1<f32>) -> Array1<f32> {
    return x;
}

fn softmax(x: Array1<f32>) -> Array1<f32> {
    let exp_a = exp(&x);
    let exp_sum_a = exp_a.sum();

    exp_a / exp_sum_a
}

fn load_mnist() -> (Vec<Array1<f32>>, Vec<Array1<f32>>) {
    // Deconstruct the returned Mnist struct.
    let NormalizedMnist {
        trn_img, trn_lbl, ..
    } = MnistBuilder::new()
        .label_format_one_hot()
        .training_set_length(TRN_SIZE)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize()
        .normalize();

    println!("{:?}", trn_lbl);

    let trn_lbl: Vec<f32> = trn_lbl.into_iter().map(|x| x as f32).collect();

    let mut trn_lbl_vec = Vec::new();

    for i in 1..50 {
        let prev = (i - 1) * 10;
        let next = i * 10;
        let sl = &trn_lbl[prev..next];
        let lbl = ndarray::arr1(sl);
        trn_lbl_vec.push(lbl);
    }

    let mut trn_img_vec = Vec::new();

    for i in 1..50 {
        let prev = (i - 1) * (ROWS * COLS) as usize;
        let next = i * (ROWS * COLS) as usize;
        let sl = &trn_img[prev..next];
        let img: Array1<f32> = ndarray::arr1(sl);
        trn_img_vec.push(img);
    }

    (trn_img_vec, trn_lbl_vec)
}

fn mean_squared_error(y: &Array1<f32>, t: &Array1<f32>) -> f32 {
    0.5 * (y - t).map(|x| x.powi(2)).sum()
}

fn log(y: &f32) -> f32 {
    y.ln()
}

fn cross_entropy_error<D: Dimension>(y: &Array<f32, D>, t: &Array<f32, D>) -> f32 {
    let delta = Array::from_elem(y.len(), 1e-7);

    -((delta + y).map(|x| log(x)) * t).sum()
}

fn main() {
    let network = init_network();

    let (trn_img, trn_lbl) = load_mnist();

    for (i, x) in trn_img.iter().enumerate() {
        let y = forward(&network, &x);

        let mse = mean_squared_error(&y, &trn_lbl[i]);
        let cee = cross_entropy_error(&y, &trn_lbl[i]);

        println!("y: {}, lbl: {}, mse: {}, cee: {}", y, trn_lbl[i], mse, cee);
    }
}
