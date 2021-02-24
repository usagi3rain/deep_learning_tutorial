use std::fs::File;

use csv::{Reader, StringRecord};

fn open_reader(name: &str) -> Reader<File> {
    let file_path = "csv_data/data_".to_string() + name + ".csv";
    let file = File::open(file_path).unwrap();
    let mut rdr = csv::Reader::from_reader(file);
    rdr.set_headers(StringRecord::from(vec![""]));

    rdr
}

pub fn b1() -> [f32; 50] {
    let mut rdr = open_reader("b1");
    let result = rdr.records().next();
    let record = result.unwrap().unwrap();
    let vec: Vec<f32> = record.iter().map(|x| x.parse::<f32>().unwrap()).collect();
    let mut arr = [0f32; 50];
    for (place, element) in arr.iter_mut().zip(vec.iter()) {
        *place = *element;
    }

    arr
}

pub fn b2() -> [f32; 100] {
    let mut rdr = open_reader("b2");
    let result = rdr.records().next();
    let record = result.unwrap().unwrap();
    let vec: Vec<f32> = record.iter().map(|x| x.parse::<f32>().unwrap()).collect();
    let mut arr = [0f32; 100];
    for (place, element) in arr.iter_mut().zip(vec.iter()) {
        *place = *element;
    }

    arr
}

pub fn b3() -> [f32; 10] {
    let mut rdr = open_reader("b3");
    let result = rdr.records().next();
    let record = result.unwrap().unwrap();
    let vec: Vec<f32> = record.iter().map(|x| x.parse::<f32>().unwrap()).collect();
    let mut arr = [0f32; 10];
    for (place, element) in arr.iter_mut().zip(vec.iter()) {
        *place = *element;
    }

    arr
}

pub fn w1() -> Vec<f32> {
    let mut rdr = open_reader("W1");
    let mut a: Vec<f32> = Vec::new();
    for result in rdr.records() {
        // The iterator yields Result<StringRecord, Error>, so we check the
        // error here.
        let record = result.unwrap();
        let vec: Vec<f32> = record.iter().map(|x| x.parse::<f32>().unwrap()).collect();
        let mut arr = [0f32; 50];
        for (place, element) in arr.iter_mut().zip(vec.iter()) {
            *place = *element;
        }
        a.extend_from_slice(&arr);
    }

    a
}

pub fn w2() -> Vec<f32> {
    let mut rdr = open_reader("W2");
    let mut a: Vec<f32> = Vec::new();
    for result in rdr.records() {
        // The iterator yields Result<StringRecord, Error>, so we check the
        // error here.
        let record = result.unwrap();
        let vec: Vec<f32> = record.iter().map(|x| x.parse::<f32>().unwrap()).collect();
        let mut arr = [0f32; 100];
        for (place, element) in arr.iter_mut().zip(vec.iter()) {
            *place = *element;
        }
        a.extend_from_slice(&arr);
    }

    a
}

pub fn w3() -> Vec<f32> {
    let mut rdr = open_reader("W3");
    let mut a: Vec<f32> = Vec::new();
    for result in rdr.records() {
        // The iterator yields Result<StringRecord, Error>, so we check the
        // error here.
        let record = result.unwrap();
        let vec: Vec<f32> = record.iter().map(|x| x.parse::<f32>().unwrap()).collect();
        let mut arr = [0f32; 10];
        for (place, element) in arr.iter_mut().zip(vec.iter()) {
            *place = *element;
        }
        a.extend_from_slice(&arr);
    }

    a
}
