
use ndarray::{Array, Array2, Array1};
use csv::ReaderBuilder;
use std::{error::Error, io::Write, sync::{atomic::{AtomicBool, Ordering}, Arc}, time::Duration};
use std::thread;
use linfa::dataset::Dataset;

// Neural Network implementation using linfa
struct SimpleNeuralNetwork {
    weights: Array2<f64>,
    bias: Array1<f64>,
    learning_rate: f64,
}

impl SimpleNeuralNetwork {
    fn new(input_size: usize, output_size: usize, learning_rate: f64) -> Self {
        // Initialize weights with small random values and bias with zeros
        let weights = Array2::from_shape_fn((input_size, output_size), |_| {
            rand::random::<f64>() * 0.1
        });
        let bias = Array1::zeros(output_size);
        
        SimpleNeuralNetwork {
            weights,
            bias,
            learning_rate,
        }
    }
    
    // Sigmoid activation function
    fn sigmoid(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
    
    // Forward pass
    fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let output = input.dot(&self.weights) + &self.bias;
        output.mapv(|x| self.sigmoid(x))
    }
    
    // Train the network
    fn train(&mut self, inputs: &Array2<f64>, targets: &Array1<f64>, epochs: usize) {
        for _ in 0..epochs {
            for (input, target) in inputs.outer_iter().zip(targets.iter()) {
                // Forward pass
                let output = self.forward(&input.to_owned());
                
                // Calculate error
                let error = output[0] - target;
                
                // Calculate gradient (using sigmoid derivative)
                let gradient = output[0] * (1.0 - output[0]) * error;
                
                // Update weights and bias
                self.weights = &self.weights - (self.learning_rate * gradient * &input).into_shape(self.weights.dim()).unwrap();
                self.bias[0] -= self.learning_rate * gradient;
            }
        }
    }
    
    // Predict
    fn predict(&self, input: &Array1<f64>) -> f64 {
        self.forward(input)[0]
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // Baca dataset dari string
    let data = r#"
no,tanah,penambahan_air_gram,penambahan_air_ml,kelembaban_manual,kelembaban_sensor,selisih
1,100gr,0 mL,0%,0.96%,0.96%,
2,100gr,10mL,10%,10.95%,0.95%,
3,100gr,20mL,20%,29.97%,9.97%,
4,100gr,30mL,30%,48.40%,18.40%,
5,100gr,40mL,40%,54.90%,14.90%,
6,100gr,50 mL,50%,71.00%,21.00%,
7,100gr,60mL,60%,77.70%,17.70%,
8,100gr,70mL,70%,77.98%,17.98%,
9,100gr,80mL,80%,82.54%,2.54%,
10,100gr,90mL,90%,85.35%,4.65%,
11,100gr,100mL,100%,85.45%,14.55%,
"#;

    let mut rdr = ReaderBuilder::new().from_reader(data.as_bytes());
    let mut records = Vec::new();

    for result in rdr.records() {
        let record = result?;
        let penambahan_air_ml: f64 = record[3].trim_end_matches('%').parse()?;
        let kelembaban_sensor: f64 = record[5].trim_end_matches('%').parse()?;
        records.push((penambahan_air_ml, kelembaban_sensor));
    }

    // Konversi ke Array2
    let mut features = Array2::zeros((records.len(), 1));
    let mut labels = Array::zeros(records.len());

    for (i, (penambahan_air_ml, kelembaban_sensor)) in records.iter().enumerate() {
        features[[i, 0]] = *penambahan_air_ml;
        labels[i] = *kelembaban_sensor;
    }

    // Normalize features and labels (important for neural networks)
    let max_feature = features.iter().fold(f64::MIN, |a, &b| a.max(b));
    let max_label = labels.iter().fold(f64::MIN, |a, &b| a.max(b));
    
    features = features / max_feature;
    labels = labels / max_label;

    // Bagi dataset menjadi training dan testing
    let (train, test) = Dataset::new(features.clone(), labels.clone())
        .split_with_ratio(0.8);

    // Create and train the neural network
    let running_nn = Arc::new(AtomicBool::new(true));
    let running_nn_clone = running_nn.clone();

    // Thread untuk menampilkan animasi "Training Neural Network"
    let handle_nn = thread::spawn(move || {
        let mut dots = 0;
        while running_nn_clone.load(Ordering::Relaxed) {
            print!("\rTraining Neural Network{}   ", ".".repeat(dots));
            dots = (dots + 1) % 4;
            std::io::stdout().flush().unwrap();
            thread::sleep(Duration::from_millis(500));
        }
        println!("\rTraining Neural Network selesai!        ");
    });

    let mut nn = SimpleNeuralNetwork::new(1, 1, 0.1);
    nn.train(train.records(), train.targets(), 1000);

    // Hentikan animasi setelah training selesai
    running_nn.store(false, Ordering::Relaxed);
    handle_nn.join().unwrap();

    // Make predictions
    let mut predictions = Vec::new();
    for sample in test.records().outer_iter() {
        let pred = nn.predict(&sample.to_owned()) * max_label;
        predictions.push(pred);
    }

    println!("Neural Network Predictions:");
    for (i, pred) in predictions.iter().enumerate() {
        let actual = test.targets()[i] * max_label;
        println!("Sample {}: Predicted {:.2}, Actual {:.2}", i, pred, actual);
    }

    // Calculate accuracy (for regression, we'll use MSE)
    let mse = predictions.iter()
        .enumerate()
        .map(|(i, &pred)| {
            let actual = test.targets()[i] * max_label;
            (pred - actual).powi(2)
        })
        .sum::<f64>() / predictions.len() as f64;
    
    println!("Mean Squared Error: {:.4}", mse);

    Ok(())
}