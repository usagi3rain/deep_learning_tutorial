use mnist::{MnistBuilder, NormalizedMnist};
use ndarray::prelude::*;

use rand::seq::SliceRandom;

mod two_layer_net;

const TRN_SIZE: u32 = 50_000;
const ROWS: u32 = 28;
const COLS: u32 = 28;
const SIZE: (usize, usize, usize) = ((ROWS * COLS) as usize, 50, 10);

fn load_mnist() -> Vec<(Vec<f32>, Vec<f32>)> {
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

    let trn_lbl: Vec<f32> = trn_lbl.into_iter().map(|x| x as f32).collect();

    let mut trn_vec = Vec::new();

    // let lbls = [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 0.0]];
    // let imgs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];

    for i in 0..(TRN_SIZE as usize) {
        let lbl_prev = i * SIZE.2;
        let lbl_next = (i + 1) * SIZE.2;
        let lbl = Vec::from(&trn_lbl[lbl_prev..lbl_next]);
        // let lbl: Vec<f32> = Vec::from(lbls[i]);

        let prev = i * (ROWS * COLS) as usize;
        let next = (i + 1) * (ROWS * COLS) as usize;
        let img = Vec::from(&trn_img[prev..next]);
        // let img: Vec<f32> = Vec::from(imgs[i]);

        trn_vec.push((img, lbl));
    }

    trn_vec
}

fn main() {
    let trn_vec = load_mnist();
    let iters_num = 10_000;
    let batch_size = 10;
    let learning_rate = 0.1;

    let mut network = two_layer_net::TwoLayerNet::new(SIZE, 0.01);

    let mut train_loss_list: Vec<f64> = Vec::new();

    for i in 0..iters_num {
        let mut rng = &mut rand::thread_rng();
        let batch: Vec<(Vec<f32>, Vec<f32>)> = trn_vec
            .choose_multiple(&mut rng, batch_size)
            .cloned()
            .collect();

        let mut x_batch: Vec<f64> = Vec::new();
        let mut t_batch: Vec<f64> = Vec::new();

        for trn in batch.iter() {
            x_batch.append(&mut trn.0.clone().into_iter().map(|n| n as f64).collect());
            t_batch.append(&mut trn.1.clone().into_iter().map(|n| n as f64).collect());
        }

        // println!("{:?}", batch);
        // println!("{:?}", x_batch);
        // println!("{:?}", t_batch);

        let x_batch =
            Array::from_shape_vec((batch_size, (ROWS * COLS) as usize).f(), x_batch).unwrap();
        let t_batch = Array::from_shape_vec((batch_size, SIZE.2).f(), t_batch).unwrap();

        network.gradient(&x_batch, &t_batch);
        network.update_params(learning_rate);

        let loss = network.loss(&x_batch, &t_batch);

        train_loss_list.push(loss);
        println!("loss: {}", loss);
        if i % (TRN_SIZE / batch_size as u32) == 0 {
            let acc = network.accuracy(&x_batch, &t_batch);
            println!("acc: {}", acc);
            network.print();
        }
    }
}
