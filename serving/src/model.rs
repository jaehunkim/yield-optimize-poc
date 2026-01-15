use anyhow::{Context, Result};
use core::num::NonZeroUsize;
use ndarray::Array2;
use ort::{
    ep,
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use tracing::{info, instrument};

/// Configuration for ONNX Runtime optimization
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Number of threads for intra-op parallelism (within a single operator)
    pub intra_threads: usize,
    /// Number of threads for inter-op parallelism (between operators)
    pub inter_threads: usize,
    /// Enable memory pattern optimization
    pub enable_mem_pattern: bool,
    /// Number of sessions in the pool for concurrent inference
    pub pool_size: usize,
    /// Enable XNNPACK execution provider
    pub enable_xnnpack: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            intra_threads: 1,   // Single thread optimal for small models on 7800X3D
            inter_threads: 1,   // Single inference path
            enable_mem_pattern: true,
            pool_size: 8,       // 8 sessions for concurrent requests
            enable_xnnpack: true,
        }
    }
}

/// ONNX model wrapper with session pooling for concurrent inference
///
/// Uses multiple Session instances to reduce lock contention.
/// Each session is protected by its own Mutex, and requests are
/// distributed round-robin across sessions.
pub struct DeepFMModel {
    sessions: Vec<Mutex<Session>>,
    next_session: AtomicUsize,
}

impl DeepFMModel {
    /// Load ONNX model from file with default configuration
    #[instrument(skip_all)]
    pub fn load(model_path: impl AsRef<Path>) -> Result<Self> {
        Self::load_with_config(model_path, ModelConfig::default())
    }

    /// Load ONNX model with custom configuration
    #[instrument(skip_all)]
    pub fn load_with_config(model_path: impl AsRef<Path>, config: ModelConfig) -> Result<Self> {
        let path = model_path.as_ref();
        info!("Loading ONNX model from: {}", path.display());
        info!("Config: intra_threads={}, inter_threads={}, mem_pattern={}, pool_size={}",
              config.intra_threads, config.inter_threads, config.enable_mem_pattern, config.pool_size);

        let mut sessions = Vec::with_capacity(config.pool_size);

        for i in 0..config.pool_size {
            let mut builder = Session::builder()?
                .with_intra_op_spinning(false)?;

            // Optionally enable XNNPACK execution provider
            if config.enable_xnnpack {
                let xnnpack_threads = NonZeroUsize::new(config.intra_threads).unwrap_or(NonZeroUsize::new(1).unwrap());
                let xnnpack = ep::XNNPACK::default()
                    .with_intra_op_num_threads(xnnpack_threads)
                    .build();
                builder = builder.with_execution_providers([xnnpack])?;
            }

            builder = builder
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(config.intra_threads)?
                .with_inter_threads(config.inter_threads)?;

            // Memory pattern optimization - reuses memory allocations
            if config.enable_mem_pattern {
                builder = builder.with_memory_pattern(true)?;
            }

            let session = builder
                .commit_from_file(path)
                .context("Failed to load ONNX model")?;

            sessions.push(Mutex::new(session));

            if i == 0 {
                info!("First session loaded successfully");
            }
        }

        info!("Session pool initialized with {} sessions", config.pool_size);
        Ok(Self {
            sessions,
            next_session: AtomicUsize::new(0),
        })
    }

    /// Get the next session index using round-robin
    #[inline]
    fn get_session_index(&self) -> usize {
        let idx = self.next_session.fetch_add(1, Ordering::Relaxed);
        idx % self.sessions.len()
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

        // Get session from pool (round-robin)
        let session_idx = self.get_session_index();
        let mut session = self.sessions[session_idx].lock().unwrap();
        let outputs = session.run(ort::inputs![input_value])?;

        // Extract output (single CTR value)
        let (_, output_data) = outputs[0].try_extract_tensor::<f32>()?;

        // Get first element (batch_size=1)
        Ok(output_data[0])
    }

    /// Run batched inference on multiple feature vectors
    ///
    /// # Arguments
    /// * `batch_features` - Vector of feature vectors (each 15 features)
    ///
    /// # Returns
    /// * Vector of CTR predictions (0.0 to 1.0)
    ///
    /// Note: Batched inference is more efficient than multiple single predictions.
    #[instrument(skip(self, batch_features))]
    pub fn predict_batch(&self, batch_features: Vec<Vec<f32>>) -> Result<Vec<f32>> {
        if batch_features.is_empty() {
            return Ok(vec![]);
        }

        let batch_size = batch_features.len();
        let num_features = batch_features[0].len();

        // Flatten batch into single vector
        let flat_features: Vec<f32> = batch_features.into_iter().flatten().collect();
        let input_array = Array2::from_shape_vec((batch_size, num_features), flat_features)?;
        let input_value = Value::from_array(input_array)?;

        // Get session from pool (round-robin)
        let session_idx = self.get_session_index();
        let mut session = self.sessions[session_idx].lock().unwrap();
        let outputs = session.run(ort::inputs![input_value])?;

        // Extract output (batch_size CTR values)
        let (_, output_data) = outputs[0].try_extract_tensor::<f32>()?;

        Ok(output_data.iter().copied().collect())
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
