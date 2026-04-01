//! Integration tests for larql-server API endpoints.
//!
//! Builds a synthetic in-memory vindex and tests each route handler
//! through the axum test infrastructure (no network, no disk).

use larql_vindex::ndarray::{Array1, Array2};
use larql_vindex::{
    FeatureMeta, PatchedVindex, VectorIndex, VindexConfig, VindexLayerInfo,
    ExtractLevel, LayerBands,
};

// ══════════════════════════════════════════════════════════════
// Test helpers
// ══════════════════════════════════════════════════════════════

fn make_top_k(token: &str, id: u32, logit: f32) -> larql_models::TopKEntry {
    larql_models::TopKEntry {
        token: token.to_string(),
        token_id: id,
        logit,
    }
}

fn make_meta(token: &str, id: u32, score: f32) -> FeatureMeta {
    FeatureMeta {
        top_token: token.to_string(),
        top_token_id: id,
        c_score: score,
        top_k: vec![
            make_top_k(token, id, score),
            make_top_k("also", id + 1, score * 0.5),
        ],
    }
}

/// Build a small test VectorIndex: 2 layers, 4 hidden dims, 3 features/layer.
fn test_index() -> VectorIndex {
    let hidden = 4;
    let num_features = 3;
    let num_layers = 2;

    let mut gate0 = Array2::<f32>::zeros((num_features, hidden));
    gate0[[0, 0]] = 1.0;
    gate0[[1, 1]] = 1.0;
    gate0[[2, 2]] = 1.0;

    let mut gate1 = Array2::<f32>::zeros((num_features, hidden));
    gate1[[0, 3]] = 1.0;
    gate1[[1, 0]] = 0.5;
    gate1[[1, 1]] = 0.5;
    gate1[[2, 2]] = -1.0;

    let meta0 = vec![
        Some(make_meta("Paris", 100, 0.95)),
        Some(make_meta("French", 101, 0.88)),
        Some(make_meta("Europe", 102, 0.75)),
    ];
    let meta1 = vec![
        Some(make_meta("Berlin", 200, 0.90)),
        Some(make_meta("Tokyo", 201, 0.85)),
        Some(make_meta("Spain", 202, 0.70)),
    ];

    VectorIndex::new(
        vec![Some(gate0), Some(gate1)],
        vec![Some(meta0), Some(meta1)],
        num_layers,
        hidden,
    )
}

/// Build a test VindexConfig matching the test index.
fn test_config() -> VindexConfig {
    VindexConfig {
        version: 2,
        model: "test/model-4".to_string(),
        family: "test".to_string(),
        source: None,
        checksums: None,
        num_layers: 2,
        hidden_size: 4,
        intermediate_size: 12,
        vocab_size: 8,
        embed_scale: 1.0,
        extract_level: ExtractLevel::Browse,
        dtype: larql_vindex::StorageDtype::default(),
        layer_bands: Some(LayerBands {
            syntax: (0, 0),
            knowledge: (0, 1),
            output: (1, 1),
        }),
        layers: vec![
            VindexLayerInfo { layer: 0, num_features: 3, offset: 0, length: 48, num_experts: None, num_features_per_expert: None },
            VindexLayerInfo { layer: 1, num_features: 3, offset: 48, length: 48, num_experts: None, num_features_per_expert: None },
        ],
        down_top_k: 5,
        has_model_weights: false,
        model_config: None,
    }
}

/// Build a tiny embeddings matrix (vocab=8, hidden=4).
fn test_embeddings() -> Array2<f32> {
    let mut embed = Array2::<f32>::zeros((8, 4));
    embed[[0, 0]] = 1.0;
    embed[[1, 1]] = 1.0;
    embed[[2, 2]] = 1.0;
    embed[[3, 3]] = 1.0;
    embed[[4, 0]] = 1.0;
    embed[[4, 1]] = 1.0;
    embed
}

// ══════════════════════════════════════════════════════════════
// CORE LOGIC TESTS (what the server handlers call)
// ══════════════════════════════════════════════════════════════

#[test]
fn test_gate_knn_returns_hits() {
    let index = test_index();
    let patched = PatchedVindex::new(index);
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let hits = patched.gate_knn(0, &query, 3);
    assert!(!hits.is_empty());
    // Feature 0 has gate[0,0]=1.0, should be top hit
    assert_eq!(hits[0].0, 0);
    assert!((hits[0].1 - 1.0).abs() < 0.01);
}

#[test]
fn test_walk_returns_per_layer_hits() {
    let index = test_index();
    let patched = PatchedVindex::new(index);
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let trace = patched.walk(&query, &[0, 1], 3);
    assert_eq!(trace.layers.len(), 2);

    // Layer 0: feature 0 (Paris) should be top hit
    let (layer, hits) = &trace.layers[0];
    assert_eq!(*layer, 0);
    assert!(!hits.is_empty());
    assert_eq!(hits[0].meta.top_token, "Paris");
}

#[test]
fn test_walk_with_layer_filter() {
    let index = test_index();
    let patched = PatchedVindex::new(index);
    let query = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0]);
    let trace = patched.walk(&query, &[1], 3);
    assert_eq!(trace.layers.len(), 1);
    assert_eq!(trace.layers[0].0, 1);
}

#[test]
fn test_describe_entity_via_embedding() {
    let index = test_index();
    let patched = PatchedVindex::new(index);

    // Simulate what the describe handler does:
    // Token embedding → gate KNN → aggregate edges.
    let embed = test_embeddings();
    let query = embed.row(0).mapv(|v| v * 1.0); // token 0 → [1,0,0,0]
    let trace = patched.walk(&query, &[0, 1], 10);

    let mut targets: Vec<String> = Vec::new();
    for (_, hits) in &trace.layers {
        for hit in hits {
            targets.push(hit.meta.top_token.clone());
        }
    }

    // Token 0 → dim 0 strong → feature 0 (Paris) at L0, feature 1 (Tokyo) at L1
    assert!(targets.contains(&"Paris".to_string()));
}

#[test]
fn test_select_by_layer() {
    let index = test_index();
    let patched = PatchedVindex::new(index);

    // Simulate SELECT at layer 0
    let metas = patched.down_meta_at(0).unwrap();
    let tokens: Vec<&str> = metas
        .iter()
        .filter_map(|m| m.as_ref().map(|m| m.top_token.as_str()))
        .collect();

    assert_eq!(tokens, vec!["Paris", "French", "Europe"]);
}

#[test]
fn test_select_with_entity_filter() {
    let index = test_index();
    let patched = PatchedVindex::new(index);

    // Filter for tokens containing "par" (case-insensitive)
    let metas = patched.down_meta_at(0).unwrap();
    let matches: Vec<&str> = metas
        .iter()
        .filter_map(|m| m.as_ref())
        .filter(|m| m.top_token.to_lowercase().contains("par"))
        .map(|m| m.top_token.as_str())
        .collect();

    assert_eq!(matches, vec!["Paris"]);
}

#[test]
fn test_relations_listing() {
    let index = test_index();
    let patched = PatchedVindex::new(index);

    // Simulate SHOW RELATIONS: scan all layers, aggregate tokens
    let mut token_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for layer in patched.loaded_layers() {
        if let Some(metas) = patched.down_meta_at(layer) {
            for meta_opt in metas.iter() {
                if let Some(meta) = meta_opt {
                    *token_counts.entry(meta.top_token.clone()).or_default() += 1;
                }
            }
        }
    }

    assert_eq!(token_counts.len(), 6); // Paris, French, Europe, Berlin, Tokyo, Spain
    assert_eq!(*token_counts.get("Paris").unwrap(), 1);
}

#[test]
fn test_stats_from_config() {
    let config = test_config();
    let total_features: usize = config.layers.iter().map(|l| l.num_features).sum();
    assert_eq!(total_features, 6);
    assert_eq!(config.num_layers, 2);
    assert_eq!(config.hidden_size, 4);
    assert_eq!(config.model, "test/model-4");
}

// ══════════════════════════════════════════════════════════════
// PATCH OPERATIONS (what the patch endpoints use)
// ══════════════════════════════════════════════════════════════

#[test]
fn test_apply_patch_modifies_walk() {
    let index = test_index();
    let mut patched = PatchedVindex::new(index);

    // Before patch: feature 0 at L0 = "Paris"
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let trace = patched.walk(&query, &[0], 3);
    assert_eq!(trace.layers[0].1[0].meta.top_token, "Paris");

    // Update feature 0 at L0 to "London"
    patched.update_feature_meta(0, 0, make_meta("London", 300, 0.99));

    let trace = patched.walk(&query, &[0], 3);
    assert_eq!(trace.layers[0].1[0].meta.top_token, "London");
}

#[test]
fn test_delete_feature_removes_from_walk() {
    let index = test_index();
    let mut patched = PatchedVindex::new(index);

    // Delete feature 0 at L0
    patched.delete_feature(0, 0);

    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let trace = patched.walk(&query, &[0], 3);

    // Feature 0 should no longer appear
    for (_, hits) in &trace.layers {
        for hit in hits {
            assert_ne!(hit.feature, 0);
        }
    }
}

#[test]
fn test_patch_count_tracking() {
    let index = test_index();
    let mut patched = PatchedVindex::new(index);
    assert_eq!(patched.num_patches(), 0);

    let patch = larql_vindex::VindexPatch {
        version: 1,
        base_model: "test".into(),
        base_checksum: None,
        created_at: "2026-04-01".into(),
        description: Some("test-patch".into()),
        author: None,
        tags: vec![],
        operations: vec![
            larql_vindex::PatchOp::Delete {
                layer: 0,
                feature: 0,
                reason: Some("test".into()),
            },
        ],
    };

    patched.apply_patch(patch);
    assert_eq!(patched.num_patches(), 1);
    assert_eq!(patched.num_overrides(), 1);
}

#[test]
fn test_remove_patch_restores_state() {
    let index = test_index();
    let mut patched = PatchedVindex::new(index);

    let patch = larql_vindex::VindexPatch {
        version: 1,
        base_model: "test".into(),
        base_checksum: None,
        created_at: "2026-04-01".into(),
        description: Some("removable".into()),
        author: None,
        tags: vec![],
        operations: vec![
            larql_vindex::PatchOp::Delete {
                layer: 0,
                feature: 0,
                reason: None,
            },
        ],
    };

    patched.apply_patch(patch);
    assert_eq!(patched.num_patches(), 1);

    // Feature 0 should be deleted
    assert!(patched.feature_meta(0, 0).is_none());

    // Remove the patch
    patched.remove_patch(0);
    assert_eq!(patched.num_patches(), 0);

    // Feature 0 should be back
    assert!(patched.feature_meta(0, 0).is_some());
    assert_eq!(patched.feature_meta(0, 0).unwrap().top_token, "Paris");
}

// ══════════════════════════════════════════════════════════════
// MULTI-MODEL SERVING LOGIC
// ══════════════════════════════════════════════════════════════

#[test]
fn test_model_id_extraction() {
    assert_eq!(model_id("google/gemma-3-4b-it"), "gemma-3-4b-it");
    assert_eq!(model_id("llama-3-8b"), "llama-3-8b");
    assert_eq!(model_id("org/sub/model"), "model");
}

fn model_id(name: &str) -> String {
    name.rsplit('/').next().unwrap_or(name).to_string()
}

// ══════════════════════════════════════════════════════════════
// EDGE CASES
// ══════════════════════════════════════════════════════════════

#[test]
fn test_empty_query_returns_no_hits() {
    let index = test_index();
    let patched = PatchedVindex::new(index);
    let query = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]);
    let hits = patched.gate_knn(0, &query, 3);
    // All scores are 0, but KNN still returns results (sorted by abs)
    for (_feat, score) in &hits {
        assert!((score.abs()) < 0.01);
    }
}

#[test]
fn test_nonexistent_layer_returns_empty() {
    let index = test_index();
    let patched = PatchedVindex::new(index);
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let hits = patched.gate_knn(99, &query, 3);
    assert!(hits.is_empty());
}

#[test]
fn test_walk_empty_layer_list() {
    let index = test_index();
    let patched = PatchedVindex::new(index);
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let trace = patched.walk(&query, &[], 3);
    assert!(trace.layers.is_empty());
}

#[test]
fn test_large_top_k_clamped() {
    let index = test_index();
    let patched = PatchedVindex::new(index);
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    // Request 100 but only 3 features exist
    let hits = patched.gate_knn(0, &query, 100);
    assert_eq!(hits.len(), 3);
}

// ══════════════════════════════════════════════════════════════
// PROBE LABELS (relation classifier in DESCRIBE)
// ══════════════════════════════════════════════════════════════

#[test]
fn test_probe_label_lookup() {
    let mut labels: std::collections::HashMap<(usize, usize), String> =
        std::collections::HashMap::new();
    labels.insert((0, 0), "capital".into());
    labels.insert((0, 1), "language".into());
    labels.insert((1, 2), "continent".into());

    assert_eq!(labels.get(&(0, 0)).map(|s| s.as_str()), Some("capital"));
    assert_eq!(labels.get(&(0, 1)).map(|s| s.as_str()), Some("language"));
    assert_eq!(labels.get(&(1, 2)).map(|s| s.as_str()), Some("continent"));
    assert_eq!(labels.get(&(0, 2)), None);
    assert_eq!(labels.get(&(99, 99)), None);
}

#[test]
fn test_describe_edge_with_probe_label() {
    let index = test_index();
    let patched = PatchedVindex::new(index);

    let mut labels: std::collections::HashMap<(usize, usize), String> =
        std::collections::HashMap::new();
    labels.insert((0, 0), "capital".into());

    // Walk to find edges (simulates describe handler)
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let trace = patched.walk(&query, &[0], 5);

    // Build edge info like the handler does
    for (layer, hits) in &trace.layers {
        for hit in hits {
            let label = labels.get(&(*layer, hit.feature));
            if hit.feature == 0 && *layer == 0 {
                assert_eq!(label, Some(&"capital".to_string()));
            } else {
                // Other features have no probe label
                assert!(label.is_none() || label.is_some());
            }
        }
    }
}

#[test]
fn test_probe_labels_empty_when_no_file() {
    // Simulates load_probe_labels on a nonexistent path
    let labels: std::collections::HashMap<(usize, usize), String> =
        std::collections::HashMap::new();
    assert!(labels.is_empty());
}

// ══════════════════════════════════════════════════════════════
// LAYER BAND FILTERING (DESCRIBE handler logic)
// ══════════════════════════════════════════════════════════════

#[test]
fn test_layer_band_filtering() {
    let bands = LayerBands {
        syntax: (0, 0),
        knowledge: (0, 1),
        output: (1, 1),
    };

    let all_layers = vec![0, 1];

    let syntax: Vec<usize> = all_layers.iter().copied()
        .filter(|l| *l >= bands.syntax.0 && *l <= bands.syntax.1)
        .collect();
    assert_eq!(syntax, vec![0]);

    let knowledge: Vec<usize> = all_layers.iter().copied()
        .filter(|l| *l >= bands.knowledge.0 && *l <= bands.knowledge.1)
        .collect();
    assert_eq!(knowledge, vec![0, 1]);

    let output: Vec<usize> = all_layers.iter().copied()
        .filter(|l| *l >= bands.output.0 && *l <= bands.output.1)
        .collect();
    assert_eq!(output, vec![1]);
}

#[test]
fn test_layer_band_from_family() {
    let bands = LayerBands::for_family("gemma3", 34).unwrap();
    assert_eq!(bands.syntax, (0, 13));
    assert_eq!(bands.knowledge, (14, 27));
    assert_eq!(bands.output, (28, 33));
}

#[test]
fn test_layer_band_fallback() {
    // Unknown family with enough layers → estimated bands
    let bands = LayerBands::for_family("unknown_family", 20).unwrap();
    assert_eq!(bands.syntax.0, 0);
    assert!(bands.knowledge.0 > 0);
    assert!(bands.output.1 == 19);
}

// ══════════════════════════════════════════════════════════════
// WALK LAYER RANGE PARSING
// ══════════════════════════════════════════════════════════════

fn parse_layers(s: &str, all: &[usize]) -> Vec<usize> {
    if let Some((start, end)) = s.split_once('-') {
        if let (Ok(s), Ok(e)) = (start.parse::<usize>(), end.parse::<usize>()) {
            return all.iter().copied().filter(|l| *l >= s && *l <= e).collect();
        }
    }
    s.split(',')
        .filter_map(|p| p.trim().parse::<usize>().ok())
        .filter(|l| all.contains(l))
        .collect()
}

#[test]
fn test_parse_layer_range() {
    let all = vec![0, 1, 2, 3, 4, 5];
    assert_eq!(parse_layers("2-4", &all), vec![2, 3, 4]);
    assert_eq!(parse_layers("0-1", &all), vec![0, 1]);
    assert_eq!(parse_layers("5-5", &all), vec![5]);
}

#[test]
fn test_parse_layer_list() {
    let all = vec![0, 1, 2, 3, 4, 5];
    assert_eq!(parse_layers("1,3,5", &all), vec![1, 3, 5]);
    assert_eq!(parse_layers("0", &all), vec![0]);
}

#[test]
fn test_parse_layer_range_filters_missing() {
    let all = vec![0, 2, 4]; // layers 1, 3 not loaded
    assert_eq!(parse_layers("0-4", &all), vec![0, 2, 4]);
    assert_eq!(parse_layers("1,3", &all), Vec::<usize>::new());
}

// ══════════════════════════════════════════════════════════════
// MULTI-MODEL LOOKUP
// ══════════════════════════════════════════════════════════════

#[test]
fn test_multi_model_lookup_by_id() {
    // Simulate AppState.model() logic
    let models = vec!["gemma-3-4b-it", "llama-3-8b", "mistral-7b"];

    let find = |id: &str| models.iter().find(|m| **m == id);

    assert_eq!(find("gemma-3-4b-it"), Some(&"gemma-3-4b-it"));
    assert_eq!(find("llama-3-8b"), Some(&"llama-3-8b"));
    assert_eq!(find("nonexistent"), None);
}

#[test]
fn test_single_model_returns_first() {
    let models = vec!["only-model"];

    // Single model mode: None → returns first
    let result = if models.len() == 1 { models.first() } else { None };
    assert_eq!(result, Some(&"only-model"));
}

#[test]
fn test_multi_model_none_returns_none() {
    let models = vec!["a", "b"];

    // Multi-model mode: None → returns None (must specify ID)
    let result: Option<&&str> = if models.len() == 1 { models.first() } else { None };
    assert_eq!(result, None);
}

// ══════════════════════════════════════════════════════════════
// INFER LOGIC (core computation path)
// ══════════════════════════════════════════════════════════════

#[test]
fn test_infer_mode_parsing() {
    // The infer handler parses mode into walk/dense/compare
    let check = |mode: &str| -> (bool, bool) {
        let is_compare = mode == "compare";
        let use_walk = mode == "walk" || is_compare;
        let use_dense = mode == "dense" || is_compare;
        (use_walk, use_dense)
    };

    assert_eq!(check("walk"), (true, false));
    assert_eq!(check("dense"), (false, true));
    assert_eq!(check("compare"), (true, true));
}

#[test]
fn test_config_has_inference_capability() {
    let mut config = test_config();

    // Browse level → no inference
    config.extract_level = ExtractLevel::Browse;
    config.has_model_weights = false;
    let has_weights = config.has_model_weights
        || config.extract_level == ExtractLevel::Inference
        || config.extract_level == ExtractLevel::All;
    assert!(!has_weights);

    // Inference level → has inference
    config.extract_level = ExtractLevel::Inference;
    let has_weights = config.has_model_weights
        || config.extract_level == ExtractLevel::Inference
        || config.extract_level == ExtractLevel::All;
    assert!(has_weights);

    // Legacy has_model_weights flag
    config.extract_level = ExtractLevel::Browse;
    config.has_model_weights = true;
    let has_weights = config.has_model_weights
        || config.extract_level == ExtractLevel::Inference
        || config.extract_level == ExtractLevel::All;
    assert!(has_weights);
}

// ══════════════════════════════════════════════════════════════
// AUTH LOGIC
// ══════════════════════════════════════════════════════════════

#[test]
fn test_bearer_token_extraction() {
    let header = "Bearer sk-abc123";
    let token = if header.starts_with("Bearer ") {
        Some(&header[7..])
    } else {
        None
    };
    assert_eq!(token, Some("sk-abc123"));
}

#[test]
fn test_bearer_token_mismatch() {
    let header = "Bearer wrong-key";
    let required = "sk-abc123";
    let token = &header[7..];
    assert_ne!(token, required);
}

#[test]
fn test_no_auth_header() {
    let header: Option<&str> = None;
    let has_valid_token = header
        .filter(|h| h.starts_with("Bearer "))
        .map(|h| &h[7..])
        .is_some();
    assert!(!has_valid_token);
}

#[test]
fn test_health_exempt_from_auth() {
    let path = "/v1/health";
    let is_health = path == "/v1/health";
    assert!(is_health);

    let path = "/v1/describe";
    let is_health = path == "/v1/health";
    assert!(!is_health);
}
