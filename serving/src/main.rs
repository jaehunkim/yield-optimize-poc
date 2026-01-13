mod api;
mod features;
mod model;

use anyhow::{Context, Result};
use clap::Parser;
use std::sync::Arc;
use tracing::{info, Level};
use tracing_subscriber;

use crate::api::{create_router, AppState};
use crate::features::FeatureProcessor;
use crate::model::{DeepFMModel, ModelConfig};

/// DeepFM CTR prediction serving server
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Port to bind the server to
    #[arg(short, long, default_value_t = 8080)]
    port: u16,

    /// Path to the ONNX model file
    #[arg(short, long, default_value = "models/deepfm_emb8_lr0.0001_dnn25612864_neg150_best.onnx")]
    model: String,

    /// Number of intra-op threads (parallelism within operators)
    #[arg(long, default_value_t = 4)]
    intra_threads: usize,

    /// Number of inter-op threads (parallelism between operators)
    #[arg(long, default_value_t = 1)]
    inter_threads: usize,

    /// Number of sessions in the pool for concurrent inference
    #[arg(long, default_value_t = 8)]
    pool_size: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Parse CLI arguments
    let args = Args::parse();

    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    info!("Starting DeepFM serving server...");

    // Load ONNX model with optimized configuration
    info!("Loading model from: {}", args.model);
    let config = ModelConfig {
        intra_threads: args.intra_threads,
        inter_threads: args.inter_threads,
        enable_mem_pattern: true,
        pool_size: args.pool_size,
    };
    let model = DeepFMModel::load_with_config(&args.model, config)
        .context("Failed to load ONNX model")?;
    info!("Model loaded successfully");

    // Load feature processor (for now, create a dummy one)
    // TODO: Load from feature_info.json in Phase 2.2
    let feature_info = features::FeatureInfo {
        sparse_features: vec![
            "weekday".to_string(),
            "hour".to_string(),
            "region".to_string(),
            "city".to_string(),
            "adexchange".to_string(),
            "domain".to_string(),
            "slotid".to_string(),
            "slotwidth".to_string(),
            "slotheight".to_string(),
            "slotvisibility".to_string(),
            "slotformat".to_string(),
            "creative".to_string(),
            "user_tag".to_string(),
        ],
        // 2 dense features (slotwidth_norm, slotheight_norm placeholders)
        dense_features: vec!["slotwidth_norm".to_string(), "slotheight_norm".to_string()],
        feature_dims: Default::default(),
        feature_vocabs: Default::default(),
    };
    let feature_processor = FeatureProcessor::new(feature_info);

    // Create application state
    let state = AppState {
        model: Arc::new(model),
        feature_processor: Arc::new(feature_processor),
    };

    // Create router
    let app = create_router(state);

    // Start server
    let addr = format!("0.0.0.0:{}", args.port);
    info!("Server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .context("Failed to bind server")?;

    axum::serve(listener, app)
        .await
        .context("Server error")?;

    Ok(())
}
