use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::Serialize;
use std::sync::Arc;
use std::time::Instant;
use tracing::{error, info, instrument};

use crate::features::{AdRequest, FeatureProcessor};
use crate::model::DeepFMModel;

/// Shared application state
#[derive(Clone)]
pub struct AppState {
    pub model: Arc<DeepFMModel>,
    pub feature_processor: Arc<FeatureProcessor>,
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

/// Create API router
pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health_check))
        .route("/predict", post(predict))
        .route("/predict_raw", post(predict_raw))
        .route("/predict_batch", post(predict_batch))
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
