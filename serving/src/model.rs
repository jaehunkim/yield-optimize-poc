use anyhow::{Context, Result};
use ndarray::Array2;
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
};
use std::path::Path;
use std::sync::Mutex;
use tracing::{info, instrument};

/// ONNX model wrapper for DeepFM inference
///
/// Note: ort 2.0.0-rc.11 requires mutable access to session.run().
/// We use Mutex to serialize inference requests within a single process.
/// For high throughput, run multiple processes behind a load balancer.
pub struct DeepFMModel {
    session: Mutex<Session>,
}

impl DeepFMModel {
    /// Load ONNX model from file
    #[instrument(skip_all)]
    pub fn load(model_path: impl AsRef<Path>) -> Result<Self> {
        let path = model_path.as_ref();
        info!("Loading ONNX model from: {}", path.display());

        // Build session with CPU-only execution (default)
        // Note: ONNX Runtime uses CPU execution provider by default.
        // GPU would require explicitly adding CUDA/TensorRT providers.
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?  // Single thread per inference for best latency
            .commit_from_file(path)
            .context("Failed to load ONNX model")?;

        info!("Model loaded successfully");
        Ok(Self {
            session: Mutex::new(session),
        })
    }

    /// Run inference on input features
    ///
    /// # Arguments
    /// * `features` - Input feature vector (15 features for DeepFM)
    ///
    /// # Returns
    /// * CTR prediction (0.0 to 1.0)
    ///
    /// Note: This method is thread-safe and can be called concurrently.
    #[instrument(skip(self, features))]
    pub fn predict(&self, features: Vec<f32>) -> Result<f32> {
        // Create input tensor (batch_size=1, num_features=15)
        let num_features = features.len();
        let input_array = Array2::from_shape_vec((1, num_features), features)?;
        let input_value = Value::from_array(input_array)?;

        // Run inference with mutex lock (required by ort 2.0.0-rc.11 API)
        let mut session = self.session.lock().unwrap();
        let outputs = session.run(ort::inputs![input_value])?;

        // Extract output (single CTR value)
        let (_, output_data) = outputs[0].try_extract_tensor::<f32>()?;

        // Get first element (batch_size=1)
        Ok(output_data[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires ONNX model file
    fn test_model_loading() {
        let model_path = "models/deepfm_emb8_lr0.0001_dnn25612864_neg150_best.onnx";
        let model = DeepFMModel::load(model_path);
        assert!(model.is_ok());
    }

    #[test]
    #[ignore] // Requires ONNX model file
    fn test_dummy_inference() {
        let model_path = "models/deepfm_emb8_lr0.0001_dnn25612864_neg150_best.onnx";
        let model = DeepFMModel::load(model_path).unwrap();

        // Dummy input (15 features)
        let features = vec![0.0; 15];
        let result = model.predict(features);

        assert!(result.is_ok());
        let ctr = result.unwrap();
        assert!(ctr >= 0.0 && ctr <= 1.0, "CTR should be between 0 and 1");
    }
}
