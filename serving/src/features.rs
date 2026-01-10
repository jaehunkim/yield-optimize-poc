use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Feature preprocessor for DeepFM model
///
/// Converts raw ad request data into model input format
pub struct FeatureProcessor {
    feature_info: FeatureInfo,
}

/// Feature metadata loaded from feature_info.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureInfo {
    pub sparse_features: Vec<String>,
    pub dense_features: Vec<String>,
    pub feature_dims: HashMap<String, usize>,
    pub feature_vocabs: HashMap<String, HashMap<String, usize>>,
}

/// Raw ad request data (iPinYou format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdRequest {
    pub weekday: u8,
    pub hour: u8,
    pub region: u16,
    pub city: u16,
    pub adexchange: u8,
    pub domain: String,
    pub slotid: String,
    pub slotwidth: u16,
    pub slotheight: u16,
    pub slotvisibility: String,
    pub slotformat: String,
    pub creative: String,
    pub user_tag: String,
}

impl FeatureProcessor {
    /// Create a new feature processor from feature_info.json
    pub fn new(feature_info: FeatureInfo) -> Self {
        Self { feature_info }
    }

    /// Load feature info from JSON file
    #[allow(dead_code)]
    pub fn load_from_json(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .context("Failed to read feature_info.json")?;
        let feature_info: FeatureInfo = serde_json::from_str(&content)
            .context("Failed to parse feature_info.json")?;
        Ok(Self::new(feature_info))
    }

    /// Process raw ad request into model input features
    ///
    /// # Returns
    /// Vec<f32> with 15 features ready for model inference
    pub fn process(&self, request: &AdRequest) -> Result<Vec<f32>> {
        let mut features = Vec::with_capacity(15);

        // Sparse features: map categorical values to indices
        for feature_name in &self.feature_info.sparse_features {
            let value = match feature_name.as_str() {
                "weekday" => request.weekday.to_string(),
                "hour" => request.hour.to_string(),
                "region" => request.region.to_string(),
                "city" => request.city.to_string(),
                "adexchange" => request.adexchange.to_string(),
                "domain" => request.domain.clone(),
                "slotid" => request.slotid.clone(),
                "slotwidth" => request.slotwidth.to_string(),
                "slotheight" => request.slotheight.to_string(),
                "slotvisibility" => request.slotvisibility.clone(),
                "slotformat" => request.slotformat.clone(),
                "creative" => request.creative.clone(),
                "user_tag" => request.user_tag.clone(),
                _ => "0".to_string(),
            };

            // Look up index in vocabulary (default to 0 if not found)
            let idx = self.feature_info
                .feature_vocabs
                .get(feature_name)
                .and_then(|vocab| vocab.get(&value))
                .copied()
                .unwrap_or(0);

            features.push(idx as f32);
        }

        // Dense features: normalized values (placeholder for now)
        // TODO: Implement actual normalization based on feature_info
        for _ in &self.feature_info.dense_features {
            features.push(0.0);
        }

        Ok(features)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ad_request_serialization() {
        let request = AdRequest {
            weekday: 4,
            hour: 15,
            region: 1,
            city: 1,
            adexchange: 1,
            domain: "3d78fcc01ed2eb8c".to_string(),
            slotid: "mm_10067_282".to_string(),
            slotwidth: 728,
            slotheight: 90,
            slotvisibility: "FirstView".to_string(),
            slotformat: "Banner".to_string(),
            creative: "other".to_string(),
            user_tag: "10006,10024,10110,13042".to_string(),
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"weekday\":4"));
    }
}
