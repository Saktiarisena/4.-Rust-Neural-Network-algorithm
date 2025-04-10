## Gambaran Program
Program ini mengimplementasikan neural network sederhana dari scratch menggunakan Rust untuk memprediksi kelembaban tanah berdasarkan penambahan air. Program menunjukkan:

1. Pembuatan neural network dengan 1 layer
2. Proses training dengan backpropagation
3. Prediksi pada data test
4. Perhitungan Mean Squared Error (MSE)

## angkah 1: Setup Proyek
1. Buat proyek baru:
```bash
cargo new rust-neural-network
cd rust-neural-network
```

2. Tambahkan dependensi ke `Cargo.toml`:
```toml
[dependencies]
[dependencies]
linfa = "0.7.1"
linfa-svm = "0.7.2"
linfa-nn = "0.7.1"
linfa-clustering = "0.7.1"
ndarray = "0.15.6"
csv = "1.1"
rand = "0.9.0"
plotters = "0.3.0"
```

## Langkah 2: Struktur Neural Network
Program ini memiliki struct utama `SimpleNeuralNetwork` dengan komponen:

```rust
struct SimpleNeuralNetwork {
    weights: Array2<f64>,    // Bobot jaringan
    bias: Array1<f64>,       // Bias
    learning_rate: f64,      // Learning rate
}
```

## Langkah 3: Persiapan Data
1. Baca data dari string CSV:
```rust
let data = r#"
no,tanah,penambahan_air_gram,penambahan_air_ml,kelembaban_manual,kelembaban_sensor,selisih
1,100gr,0 mL,0%,0.96%,0.96%,
...
"#;
```

2. Konversi ke format Array:
```rust
let mut features = Array2::zeros((records.len(), 1));
let mut labels = Array::zeros(records.len());
```

3. Normalisasi data (penting untuk neural network):
```rust
features = features / max_feature;
labels = labels / max_label;
```

## Langkah 4: Implementasi Neural Network
1. Inisialisasi network:
```rust
fn new(input_size: usize, output_size: usize, learning_rate: f64) -> Self {
    let weights = Array2::from_shape_fn((input_size, output_size), |_| {
        rand::random::<f64>() * 0.1  // Bobot awal kecil
    });
    let bias = Array1::zeros(output_size);
    // ...
}
```

2. Fungsi aktivasi sigmoid:
```rust
fn sigmoid(&self, x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
```

3. Forward pass:
```rust
fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
    input.dot(&self.weights) + &self.bias
}
```

## Langkah 5: Proses Training
1. Loop training dengan animasi loading:
```rust
let running_nn = Arc::new(AtomicBool::new(true));
// Thread animasi loading...
```

2. Backpropagation:
```rust
// Hitung error
let error = output[0] - target;

// Hitung gradient
let gradient = output[0] * (1.0 - output[0]) * error;

// Update bobot dan bias
self.weights = &self.weights - (self.learning_rate * gradient * &input).into_shape(...);
self.bias[0] -= self.learning_rate * gradient;
```

## Langkah 6: Evaluasi Model
1. Lakukan prediksi:
```rust
let pred = nn.predict(&sample.to_owned()) * max_label;  // Denormalisasi
```

2. Hitung MSE:
```rust
let mse = predictions.iter()
    .map(|&pred| (pred - actual).powi(2))
    .sum::<f64>() / predictions.len() as f64;
```

## Cara Menjalankan
1. Jalankan program:
```bash
cargo run
```

2. Output yang dihasilkan:
```
Training Neural Network....
Training Neural Network selesai!

Neural Network Predictions:
Sample 0: Predicted 12.34, Actual 10.95
Sample 1: Predicted 29.50, Actual 29.97
...

Mean Squared Error: 0.0456
```

## Optimasi dan Pengembangan
1. **Arsitektur Jaringan**:
   - Tambahkan hidden layer
   - Implementasikan aktivasi ReLU

2. **Optimasi Training**:
   - Tambahkan momentum
   - Implementasikan learning rate decay

3. **Validasi Model**:
   - Gunakan k-fold cross validation
   - Tambahkan metrik evaluasi lainnya (R² score)

## Contoh Output Lengkap
```
=== Training Neural Network ===
Training Neural Network... (animasi loading)

=== Hasil Prediksi ===
Sample 0: Predicted 10.23%, Actual 10.95%
Sample 1: Predicted 29.87%, Actual 29.97%
Sample 2: Predicted 48.12%, Actual 48.40%
...

=== Evaluasi Model ===
Mean Squared Error: 0.0456
```

## Pembelajaran
Program ini menunjukkan:
1. Cara mengimplementasikan neural network dasar di Rust
2. Proses forward dan backward propagation
3. Normalisasi data untuk neural network
4. Evaluasi model regresi

## Referensi
- [Ndarray Documentation](https://docs.rs/ndarray/latest/ndarray/)
- [Neural Networks Basics](https://towardsdatascience.com/neural-networks-from-scratch-easy-vs-hard-b26ddc2e89c7)
