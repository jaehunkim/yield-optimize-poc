/// Test ONNX model loading and dummy inference
///
/// Usage:
///   cargo run --example test_model_loading

use anyhow::Result;

// Import from main binary
use deepfm_serving::model::DeepFMModel;

fn main() -> Result<()> {
    println!("=== DeepFM Model Loading Test ===\n");

    // Model path
    let model_path = "models/deepfm_emb8_lr0.0001_dnn25612864_neg150_best.onnx";
    println!("Loading model from: {}", model_path);

    // Load model
    let model = DeepFMModel::load(model_path)?;
    println!("✓ Model loaded successfully\n");

    // Test with dummy input (15 features)
    println!("Testing dummy inference (15 features)...");
    let dummy_features = vec![0.0; 15];

    let ctr = model.predict(dummy_features)?;
    println!("✓ Inference successful");
    println!("  CTR prediction: {:.6}", ctr);
    println!("  Valid range: {}", if ctr >= 0.0 && ctr <= 1.0 { "✓ Yes" } else { "✗ No" });

    println!("\n=== Test Complete ===");
    Ok(())
}
