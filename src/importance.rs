//! Feature importance computation.
//!
//! ## Primitive Foundation
//! - T1: Recursion (ρ) — traverse all split nodes
//! - T1: Mapping (μ) — feature index → accumulated importance
//! - T1: Sequence (σ) — sorted importance output

use crate::node::{DecisionTree, TreeNode};
use crate::types::FeatureImportance;

/// Compute feature importance scores based on weighted impurity decrease.
///
/// Each feature's importance = sum of (weighted impurity decrease × node fraction)
/// across all splits that use that feature. Normalized to sum to 1.0.
///
/// Returns an empty vector if the tree is not fitted.
#[must_use]
pub fn feature_importance(tree: &DecisionTree) -> Vec<FeatureImportance> {
    let Some(root) = tree.root() else {
        return Vec::new();
    };

    let n_features = tree.n_features;
    let total_samples = tree.n_samples as f64;

    if total_samples == 0.0 || n_features == 0 {
        return Vec::new();
    }

    // Accumulate importance per feature
    let mut raw_importance = vec![0.0_f64; n_features];
    accumulate_importance(root, &mut raw_importance, total_samples);

    // Normalize to sum to 1.0
    let total: f64 = raw_importance.iter().sum();
    if total > f64::EPSILON {
        for imp in &mut raw_importance {
            *imp /= total;
        }
    }

    // Build sorted result
    let mut result: Vec<FeatureImportance> = raw_importance
        .into_iter()
        .enumerate()
        .map(|(i, importance)| FeatureImportance {
            index: i,
            name: tree.feature_names.get(i).cloned(),
            importance,
        })
        .collect();

    // Sort by importance descending
    result.sort_by(|a, b| {
        b.importance
            .partial_cmp(&a.importance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    result
}

/// Recursively accumulate importance from split nodes.
fn accumulate_importance(node: &TreeNode, importance: &mut [f64], total_samples: f64) {
    match node {
        TreeNode::Leaf { .. } => {}
        TreeNode::Split {
            feature_index,
            left,
            right,
            samples,
            impurity_decrease,
            ..
        } => {
            // Weighted impurity decrease: (n_samples / total) × decrease
            let weight = *samples as f64 / total_samples;
            if let Some(slot) = importance.get_mut(*feature_index) {
                *slot += weight * impurity_decrease;
            }
            accumulate_importance(left, importance, total_samples);
            accumulate_importance(right, importance, total_samples);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::fit;
    use crate::types::{Feature, TreeConfig};

    #[test]
    fn importance_unfitted_returns_empty() {
        let tree = DecisionTree::new(TreeConfig::default());
        let imp = feature_importance(&tree);
        assert!(imp.is_empty());
    }

    #[test]
    fn importance_sums_to_one() {
        let data = vec![
            vec![Feature::Continuous(0.1), Feature::Continuous(0.5)],
            vec![Feature::Continuous(0.2), Feature::Continuous(0.4)],
            vec![Feature::Continuous(0.8), Feature::Continuous(0.6)],
            vec![Feature::Continuous(0.9), Feature::Continuous(0.7)],
        ];
        let labels = vec!["A".into(), "A".into(), "B".into(), "B".into()];
        let result = fit(&data, &labels, TreeConfig::default());
        assert!(result.is_ok());
        let tree = result
            .ok()
            .unwrap_or_else(|| DecisionTree::new(TreeConfig::default()));

        let imp = feature_importance(&tree);
        let total: f64 = imp.iter().map(|fi| fi.importance).sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn importance_sorted_descending() {
        let data = vec![
            vec![Feature::Continuous(0.1), Feature::Continuous(0.5)],
            vec![Feature::Continuous(0.2), Feature::Continuous(0.4)],
            vec![Feature::Continuous(0.8), Feature::Continuous(0.6)],
            vec![Feature::Continuous(0.9), Feature::Continuous(0.7)],
        ];
        let labels = vec!["A".into(), "A".into(), "B".into(), "B".into()];
        let result = fit(&data, &labels, TreeConfig::default());
        assert!(result.is_ok());
        let tree = result
            .ok()
            .unwrap_or_else(|| DecisionTree::new(TreeConfig::default()));

        let imp = feature_importance(&tree);
        for window in imp.windows(2) {
            assert!(window[0].importance >= window[1].importance);
        }
    }

    #[test]
    fn single_feature_gets_full_importance() {
        let data = vec![
            vec![Feature::Continuous(0.1)],
            vec![Feature::Continuous(0.9)],
        ];
        let labels = vec!["A".into(), "B".into()];
        let result = fit(&data, &labels, TreeConfig::default());
        assert!(result.is_ok());
        let tree = result
            .ok()
            .unwrap_or_else(|| DecisionTree::new(TreeConfig::default()));

        let imp = feature_importance(&tree);
        assert_eq!(imp.len(), 1);
        assert!((imp[0].importance - 1.0).abs() < 1e-10);
    }
}
