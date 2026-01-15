use anyhow::Result;
use deepfm_serving::model::{DeepFMModel, ModelConfig};
use std::env;
use std::fs::File;
use std::io::Read;
use std::time::Instant;

fn load_sample_features(path: &str, num_features: usize) -> Result<Vec<f32>> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let mut features = Vec::with_capacity(num_features);
    for j in 0..num_features {
        let offset = j * 4;
        let bytes = [
            buffer[offset],
            buffer[offset + 1],
            buffer[offset + 2],
            buffer[offset + 3],
        ];
        features.push(f32::from_le_bytes(bytes));
    }

    Ok(features)
}

fn main() -> Result<()> {
    // Parse CLI args: --no-xnnpack, --int8, --threads=N
    let args: Vec<String> = env::args().collect();
    let enable_xnnpack = !args.iter().any(|a| a == "--no-xnnpack");
    let use_int8 = args.iter().any(|a| a == "--int8");
    let intra_threads: usize = args.iter()
        .find(|a| a.starts_with("--threads="))
        .and_then(|a| a.split('=').nth(1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    // Load the 71M AutoInt model (emb768, att8x12)
    let model_path = if use_int8 {
        "training/models/autoint_emb768_att8x12_dnn20481024512_best_int8_dynamic.onnx"
    } else {
        "training/models/autoint_emb768_att8x12_dnn20481024512_best.onnx"
    };

    println!("Loading model: {}", model_path);
    println!("XNNPACK: {}, INT8: {}, Threads: {}",
        if enable_xnnpack { "enabled" } else { "disabled" },
        if use_int8 { "yes" } else { "no" },
        intra_threads);

    // Use single session for benchmark
    let config = ModelConfig {
        intra_threads,
        inter_threads: 1,
        enable_mem_pattern: true,
        pool_size: 1,
        enable_xnnpack,
    };

    let model = DeepFMModel::load_with_config(model_path, config)?;
    println!("Model loaded successfully!\n");

    // Load real sample features (raw binary)
    let sample_path = "serving/models/sample_features.bin";
    println!("Loading sample features: {}", sample_path);
    let features = load_sample_features(sample_path, 15)?;
    println!("Features: {:?}\n", features);

    // Warmup iterations
    println!("=== Warmup (100 iterations) ===");
    for _ in 0..100 {
        let _ = model.predict(features.clone())?;
    }
    println!("Warmup complete\n");

    // Benchmark single inference
    println!("=== Single Inference Benchmark ===");
    let iterations = 1000;
    let mut latencies: Vec<f64> = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let start = Instant::now();
        let _result = model.predict(features.clone())?;
        let elapsed = start.elapsed();
        latencies.push(elapsed.as_nanos() as f64 / 1000.0); // Convert to microseconds
    }

    // Calculate statistics
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let sum: f64 = latencies.iter().sum();
    let mean = sum / iterations as f64;
    let min = latencies[0];
    let max = latencies[iterations - 1];
    let p50 = latencies[iterations / 2];
    let p90 = latencies[(iterations as f64 * 0.9) as usize];
    let p95 = latencies[(iterations as f64 * 0.95) as usize];
    let p99 = latencies[(iterations as f64 * 0.99) as usize];

    println!("Iterations: {}", iterations);
    println!("\nLatency Statistics (microseconds):");
    println!("  Mean:  {:>10.2} µs", mean);
    println!("  Min:   {:>10.2} µs", min);
    println!("  Max:   {:>10.2} µs", max);
    println!("  P50:   {:>10.2} µs", p50);
    println!("  P90:   {:>10.2} µs", p90);
    println!("  P95:   {:>10.2} µs", p95);
    println!("  P99:   {:>10.2} µs", p99);

    println!("\nLatency Statistics (milliseconds):");
    println!("  Mean:  {:>10.3} ms", mean / 1000.0);
    println!("  P50:   {:>10.3} ms", p50 / 1000.0);
    println!("  P95:   {:>10.3} ms", p95 / 1000.0);
    println!("  P99:   {:>10.3} ms", p99 / 1000.0);

    println!("\nThroughput: {:.0} predictions/sec", 1_000_000.0 / mean);

    Ok(())
}
