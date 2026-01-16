use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tracing::{error, info, instrument};

use crate::features::{AdRequest, FeatureProcessor};
use crate::model::{DeepFMModel, OnnxModel};

/// Shared application state
#[derive(Clone)]
pub struct AppState {
    pub model: Arc<DeepFMModel>,
    pub feature_processor: Arc<FeatureProcessor>,
    /// Stage 1 model for Multi-Stage ranking (DeepFM - fast filtering)
    pub stage1_model: Option<Arc<OnnxModel>>,
    /// Stage 2 model for Multi-Stage ranking (AutoInt - precise ranking)
    pub stage2_model: Option<Arc<OnnxModel>>,
}

/// Health check response
#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}

/// Prediction response
#[derive(Serialize)]
pub struct PredictionResponse {
    pub ctr: f32,
    pub latency_ms: f64,
}

/// Raw feature vector request (for benchmarking)
#[derive(serde::Deserialize)]
pub struct RawFeatureRequest {
    pub features: Vec<f32>,
}

/// Batch feature vector request (for batched inference)
#[derive(serde::Deserialize)]
pub struct BatchFeatureRequest {
    pub batch: Vec<Vec<f32>>,
}

/// Batch prediction response
#[derive(Serialize)]
pub struct BatchPredictionResponse {
    pub predictions: Vec<f32>,
    pub batch_size: usize,
    pub latency_ms: f64,
}

/// Error response
#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

/// Rank request for Multi-Stage ranking
#[derive(Deserialize)]
pub struct RankRequest {
    /// Batch of feature vectors (each with 15 features)
    pub candidates: Vec<Vec<f32>>,
    /// Final top-k to return
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Stage 1 top-k (for multi-stage)
    #[serde(default = "default_stage1_k")]
    pub stage1_k: usize,
    /// Use multi-stage ranking (true) or single-stage (false)
    #[serde(default = "default_multi_stage")]
    pub multi_stage: bool,
}

fn default_top_k() -> usize { 10 }
fn default_stage1_k() -> usize { 100 }
fn default_multi_stage() -> bool { true }

/// Single ranking result
#[derive(Serialize)]
pub struct RankItem {
    pub index: usize,
    pub ctr: f32,
    pub rank: usize,
}

/// Rank response
#[derive(Serialize)]
pub struct RankResponse {
    pub rankings: Vec<RankItem>,
    pub latency: RankLatency,
    pub stats: RankStats,
}

/// Latency breakdown for ranking
#[derive(Serialize)]
pub struct RankLatency {
    pub stage1_ms: f64,
    pub stage2_ms: f64,
    pub total_ms: f64,
}

/// Stats for ranking
#[derive(Serialize)]
pub struct RankStats {
    pub input: usize,
    pub after_stage1: usize,
    pub output: usize,
    pub mode: String,
}

/// Create API router
pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health_check))
        .route("/predict", post(predict))
        .route("/predict_raw", post(predict_raw))
        .route("/predict_batch", post(predict_batch))
        .route("/rank", post(rank))
        .with_state(state)
}

/// Health check endpoint
#[instrument]
async fn health_check() -> impl IntoResponse {
    info!("Health check requested");
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

/// CTR prediction endpoint
#[instrument(skip(state, request))]
async fn predict(
    State(state): State<AppState>,
    Json(request): Json<AdRequest>,
) -> Result<Json<PredictionResponse>, (StatusCode, Json<ErrorResponse>)> {
    let start = Instant::now();

    // Preprocess features
    let features = state
        .feature_processor
        .process(&request)
        .map_err(|e| {
            error!("Feature processing failed: {}", e);
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!("Feature processing failed: {}", e),
                }),
            )
        })?;

    // Run inference
    let ctr = state.model.predict(features).map_err(|e| {
        error!("Model inference failed: {}", e);
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("Model inference failed: {}", e),
            }),
        )
    })?;

    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Debug-level logging for benchmark performance
    tracing::debug!("Prediction completed: CTR={:.6}, latency={:.2}ms", ctr, latency_ms);

    Ok(Json(PredictionResponse { ctr, latency_ms }))
}

/// CTR prediction endpoint with raw feature vector (for benchmarking)
#[instrument(skip(state, request))]
async fn predict_raw(
    State(state): State<AppState>,
    Json(request): Json<RawFeatureRequest>,
) -> Result<Json<PredictionResponse>, (StatusCode, Json<ErrorResponse>)> {
    let start = Instant::now();

    // Validate feature count (12 sparse + 3 dense = 15 features)
    if request.features.len() != 15 {
        error!("Invalid feature count: expected 15, got {}", request.features.len());
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!("Expected 15 features, got {}", request.features.len()),
            }),
        ));
    }

    // Run inference
    let ctr = state.model.predict(request.features).map_err(|e| {
        error!("Model inference failed: {}", e);
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("Model inference failed: {}", e),
            }),
        )
    })?;

    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Debug-level logging for benchmark performance
    tracing::debug!("Prediction completed: CTR={:.6}, latency={:.2}ms", ctr, latency_ms);

    Ok(Json(PredictionResponse { ctr, latency_ms }))
}

/// Batched CTR prediction endpoint (for benchmarking batched inference)
#[instrument(skip(state, request))]
async fn predict_batch(
    State(state): State<AppState>,
    Json(request): Json<BatchFeatureRequest>,
) -> Result<Json<BatchPredictionResponse>, (StatusCode, Json<ErrorResponse>)> {
    let start = Instant::now();

    // Validate batch
    if request.batch.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Empty batch".to_string(),
            }),
        ));
    }

    // Validate feature count for each item
    for (i, features) in request.batch.iter().enumerate() {
        if features.len() != 15 {
            error!("Invalid feature count at index {}: expected 15, got {}", i, features.len());
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!("Expected 15 features at index {}, got {}", i, features.len()),
                }),
            ));
        }
    }

    let batch_size = request.batch.len();

    // Run batched inference
    let predictions = state.model.predict_batch(request.batch).map_err(|e| {
        error!("Batch inference failed: {}", e);
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("Batch inference failed: {}", e),
            }),
        )
    })?;

    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

    tracing::debug!(
        "Batch prediction completed: batch_size={}, latency={:.2}ms, per_item={:.3}ms",
        batch_size,
        latency_ms,
        latency_ms / batch_size as f64
    );

    Ok(Json(BatchPredictionResponse {
        predictions,
        batch_size,
        latency_ms,
    }))
}

/// Multi-Stage ranking endpoint
///
/// Supports two modes:
/// - Single-Stage: AutoInt (12MB) processes all candidates
/// - Multi-Stage: DeepFM (18KB) filters to top-K, then AutoInt (12MB) ranks
#[instrument(skip(state, request))]
async fn rank(
    State(state): State<AppState>,
    Json(request): Json<RankRequest>,
) -> Result<Json<RankResponse>, (StatusCode, Json<ErrorResponse>)> {
    let total_start = Instant::now();

    // Validate input
    if request.candidates.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Empty candidates".to_string(),
            }),
        ));
    }

    let input_size = request.candidates.len();

    // Check if models are loaded
    let stage2_model = state.stage2_model.as_ref().ok_or_else(|| {
        error!("Stage 2 model (AutoInt) not loaded");
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "Stage 2 model (AutoInt) not loaded. Start server with --autoint-model".to_string(),
            }),
        )
    })?;

    let (rankings, stage1_ms, stage2_ms, after_stage1, mode) = if request.multi_stage {
        // Multi-Stage: DeepFM -> AutoInt
        let stage1_model = state.stage1_model.as_ref().ok_or_else(|| {
            error!("Stage 1 model (DeepFM) not loaded");
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorResponse {
                    error: "Stage 1 model (DeepFM) not loaded. Start server with --deepfm-model".to_string(),
                }),
            )
        })?;

        // Stage 1: DeepFM for fast filtering
        let stage1_start = Instant::now();
        let stage1_scores = stage1_model.predict_batch(request.candidates.clone()).map_err(|e| {
            error!("Stage 1 inference failed: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Stage 1 inference failed: {}", e),
                }),
            )
        })?;
        let stage1_ms = stage1_start.elapsed().as_secs_f64() * 1000.0;

        // Get top-K indices from Stage 1
        let stage1_k = request.stage1_k.min(input_size);
        let mut indexed_scores: Vec<(usize, f32)> = stage1_scores
            .into_iter()
            .enumerate()
            .collect();
        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_k_indices: Vec<usize> = indexed_scores.iter().take(stage1_k).map(|(i, _)| *i).collect();

        // Extract top-K candidates for Stage 2
        let stage2_candidates: Vec<Vec<f32>> = top_k_indices
            .iter()
            .map(|&i| request.candidates[i].clone())
            .collect();

        // Stage 2: AutoInt for precise ranking
        let stage2_start = Instant::now();
        let stage2_scores = stage2_model.predict_batch(stage2_candidates).map_err(|e| {
            error!("Stage 2 inference failed: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Stage 2 inference failed: {}", e),
                }),
            )
        })?;
        let stage2_ms = stage2_start.elapsed().as_secs_f64() * 1000.0;

        // Combine indices with Stage 2 scores and sort
        let mut final_scores: Vec<(usize, f32)> = top_k_indices
            .into_iter()
            .zip(stage2_scores.into_iter())
            .collect();
        final_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take final top-k
        let rankings: Vec<RankItem> = final_scores
            .into_iter()
            .take(request.top_k)
            .enumerate()
            .map(|(rank, (index, ctr))| RankItem {
                index,
                ctr,
                rank: rank + 1,
            })
            .collect();

        (rankings, stage1_ms, stage2_ms, stage1_k, "multi-stage".to_string())
    } else {
        // Single-Stage: AutoInt only
        let stage2_start = Instant::now();
        let scores = stage2_model.predict_batch(request.candidates).map_err(|e| {
            error!("Single-stage inference failed: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Single-stage inference failed: {}", e),
                }),
            )
        })?;
        let stage2_ms = stage2_start.elapsed().as_secs_f64() * 1000.0;

        // Sort and take top-k
        let mut indexed_scores: Vec<(usize, f32)> = scores
            .into_iter()
            .enumerate()
            .collect();
        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let rankings: Vec<RankItem> = indexed_scores
            .into_iter()
            .take(request.top_k)
            .enumerate()
            .map(|(rank, (index, ctr))| RankItem {
                index,
                ctr,
                rank: rank + 1,
            })
            .collect();

        (rankings, 0.0, stage2_ms, input_size, "single-stage".to_string())
    };

    let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;

    info!(
        "Rank completed: mode={}, input={}, after_stage1={}, output={}, latency={:.2}ms",
        mode, input_size, after_stage1, rankings.len(), total_ms
    );

    Ok(Json(RankResponse {
        rankings,
        latency: RankLatency {
            stage1_ms,
            stage2_ms,
            total_ms,
        },
        stats: RankStats {
            input: input_size,
            after_stage1,
            output: request.top_k.min(after_stage1),
            mode,
        },
    }))
}
