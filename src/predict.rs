//! Prediction with confidence and explainability.
//!
//! ## Primitive Foundation
//! - T1: Recursion (ρ) — traverse from root to leaf
//! - T1: Mapping (μ) — feature value → direction at each split
//! - T1: Exists (∃) — leaf prediction existence check

use crate::node::{DecisionTree, TreeNode};
use crate::types::{Direction, Feature, PredictionResult, RegressionResult, SplitDescription};

/// Error type for prediction failures.
#[derive(Debug, nexcore_error::Error)]
pub enum PredictError {
    /// Tree has not been fitted.
    #[error("tree has not been fitted")]
    NotFitted,
    /// Feature count mismatch.
    #[error("expected {expected} features, got {got}")]
    FeatureMismatch { expected: usize, got: usize },
}

/// Predict class label for a single sample.
///
/// # Errors
/// Returns `PredictError` if tree is not fitted or feature count mismatches.
pub fn predict(
    tree: &DecisionTree,
    features: &[Feature],
) -> Result<PredictionResult, PredictError> {
    if !tree.is_fitted() {
        return Err(PredictError::NotFitted);
    }
    if features.len() != tree.n_features {
        return Err(PredictError::FeatureMismatch {
            expected: tree.n_features,
            got: features.len(),
        });
    }

    let root = tree.root().ok_or(PredictError::NotFitted)?;
    let mut path = Vec::new();
    let leaf = traverse(root, features, &tree.feature_names, &mut path, 0);

    match leaf {
        TreeNode::Leaf {
            prediction,
            confidence,
            distribution,
            samples,
            ..
        } => Ok(PredictionResult {
            prediction: prediction.clone(),
            confidence: *confidence,
            class_distribution: distribution.clone(),
            leaf_samples: *samples,
            depth: path.len(),
            path,
        }),
        TreeNode::Split { .. } => Err(PredictError::NotFitted),
    }
}

/// Predict numeric value for a single sample (regression).
///
/// # Errors
/// Returns `PredictError` if tree is not fitted or feature count mismatches.
pub fn predict_regression(
    tree: &DecisionTree,
    features: &[Feature],
) -> Result<RegressionResult, PredictError> {
    if !tree.is_fitted() {
        return Err(PredictError::NotFitted);
    }
    if features.len() != tree.n_features {
        return Err(PredictError::FeatureMismatch {
            expected: tree.n_features,
            got: features.len(),
        });
    }

    let root = tree.root().ok_or(PredictError::NotFitted)?;
    let mut path = Vec::new();
    let leaf = traverse(root, features, &tree.feature_names, &mut path, 0);

    match leaf {
        TreeNode::Leaf {
            prediction,
            samples,
            impurity,
            ..
        } => {
            let value = prediction.parse::<f64>().unwrap_or(0.0);
            Ok(RegressionResult {
                prediction: value,
                variance: impurity.0,
                leaf_samples: *samples,
                depth: path.len(),
                path,
            })
        }
        TreeNode::Split { .. } => Err(PredictError::NotFitted),
    }
}

/// Predict multiple samples at once (classification).
///
/// # Errors
/// Returns `PredictError` on first failure.
pub fn predict_batch(
    tree: &DecisionTree,
    data: &[Vec<Feature>],
) -> Result<Vec<PredictionResult>, PredictError> {
    data.iter().map(|row| predict(tree, row)).collect()
}

/// Traverse the tree to find the leaf node for a given feature vector.
/// Collects the path taken for explainability.
fn traverse<'a>(
    node: &'a TreeNode,
    features: &[Feature],
    feature_names: &[String],
    path: &mut Vec<SplitDescription>,
    _depth: usize,
) -> &'a TreeNode {
    match node {
        TreeNode::Leaf { .. } => node,
        TreeNode::Split {
            feature_index,
            threshold,
            left,
            right,
            ..
        } => {
            let direction = decide_direction(&features[*feature_index], threshold);
            let feat_name = feature_names.get(*feature_index).cloned();

            path.push(SplitDescription {
                feature_index: *feature_index,
                feature_name: feat_name,
                threshold: format!("{threshold}"),
                direction,
            });

            let child = match direction {
                Direction::Left => left.as_ref(),
                Direction::Right => right.as_ref(),
            };
            traverse(child, features, feature_names, path, _depth + 1)
        }
    }
}

/// Decide which direction to go at a split node.
fn decide_direction(feature_value: &Feature, threshold: &Feature) -> Direction {
    match (feature_value, threshold) {
        (Feature::Continuous(v), Feature::Continuous(t)) => {
            if *v <= *t {
                Direction::Left
            } else {
                Direction::Right
            }
        }
        (Feature::Categorical(v), Feature::Categorical(t)) => {
            if v == t {
                Direction::Left
            } else {
                Direction::Right
            }
        }
        // Missing values go right (majority convention)
        _ => Direction::Right,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::fit;
    use crate::types::{Feature, TreeConfig};

    #[test]
    fn predict_unfitted_tree() {
        let tree = DecisionTree::new(TreeConfig::default());
        let result = predict(&tree, &[Feature::Continuous(1.0)]);
        assert!(result.is_err());
    }

    #[test]
    fn predict_feature_mismatch() {
        let data = vec![
            vec![Feature::Continuous(0.1), Feature::Continuous(0.2)],
            vec![Feature::Continuous(0.9), Feature::Continuous(0.8)],
        ];
        let labels = vec!["A".into(), "B".into()];
        let tree = fit(&data, &labels, TreeConfig::default());
        assert!(tree.is_ok());
        let tree = tree.ok();
        assert!(tree.is_some());
        let tree = tree.unwrap_or_else(|| DecisionTree::new(TreeConfig::default()));

        // Wrong number of features
        let result = predict(&tree, &[Feature::Continuous(0.5)]);
        assert!(result.is_err());
    }

    #[test]
    fn predict_simple_classification() {
        let data = vec![
            vec![Feature::Continuous(0.1)],
            vec![Feature::Continuous(0.2)],
            vec![Feature::Continuous(0.3)],
            vec![Feature::Continuous(0.7)],
            vec![Feature::Continuous(0.8)],
            vec![Feature::Continuous(0.9)],
        ];
        let labels = vec![
            "A".into(),
            "A".into(),
            "A".into(),
            "B".into(),
            "B".into(),
            "B".into(),
        ];
        let tree = fit(&data, &labels, TreeConfig::default());
        assert!(tree.is_ok());
        let tree = tree
            .ok()
            .unwrap_or_else(|| DecisionTree::new(TreeConfig::default()));

        // Should predict "A" for low values
        let result = predict(&tree, &[Feature::Continuous(0.15)]);
        assert!(result.is_ok());
        let pred = result.ok().unwrap_or_else(|| PredictionResult {
            prediction: String::new(),
            confidence: crate::types::Confidence::NONE,
            class_distribution: Vec::new(),
            leaf_samples: 0,
            depth: 0,
            path: Vec::new(),
        });
        assert_eq!(pred.prediction, "A");

        // Should predict "B" for high values
        let result = predict(&tree, &[Feature::Continuous(0.85)]);
        assert!(result.is_ok());
        let pred = result.ok().unwrap_or_else(|| PredictionResult {
            prediction: String::new(),
            confidence: crate::types::Confidence::NONE,
            class_distribution: Vec::new(),
            leaf_samples: 0,
            depth: 0,
            path: Vec::new(),
        });
        assert_eq!(pred.prediction, "B");
    }

    #[test]
    fn predict_returns_path() {
        let data = vec![
            vec![Feature::Continuous(0.1)],
            vec![Feature::Continuous(0.9)],
        ];
        let labels = vec!["A".into(), "B".into()];
        let tree = fit(&data, &labels, TreeConfig::default());
        assert!(tree.is_ok());
        let tree = tree
            .ok()
            .unwrap_or_else(|| DecisionTree::new(TreeConfig::default()));

        let result = predict(&tree, &[Feature::Continuous(0.05)]);
        assert!(result.is_ok());
        let pred = result.ok().unwrap_or_else(|| PredictionResult {
            prediction: String::new(),
            confidence: crate::types::Confidence::NONE,
            class_distribution: Vec::new(),
            leaf_samples: 0,
            depth: 0,
            path: Vec::new(),
        });
        // Should have at least one split in the path
        assert!(!pred.path.is_empty());
    }

    #[test]
    fn predict_batch_works() {
        let data = vec![
            vec![Feature::Continuous(0.1)],
            vec![Feature::Continuous(0.9)],
        ];
        let labels = vec!["A".into(), "B".into()];
        let tree = fit(&data, &labels, TreeConfig::default());
        assert!(tree.is_ok());
        let tree = tree
            .ok()
            .unwrap_or_else(|| DecisionTree::new(TreeConfig::default()));

        let test_data = vec![
            vec![Feature::Continuous(0.05)],
            vec![Feature::Continuous(0.95)],
        ];
        let results = predict_batch(&tree, &test_data);
        assert!(results.is_ok());
    }
}
