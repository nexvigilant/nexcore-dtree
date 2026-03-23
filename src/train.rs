//! CART training: recursive binary splitting.
//!
//! ## Primitive Foundation
//! - T1: Recursion (ρ) — `build_node` recurses left/right until leaf conditions
//! - T1: Mapping (μ) — feature index → candidate thresholds → best split
//! - T1: Sequence (σ) — sorted unique values for threshold candidates
//! - T1: State (ς) — accumulates class counts, tracks best impurity decrease

use crate::criterion::{Mse, SplitCriterion, make_criterion, split_info};
use crate::node::{DecisionTree, TreeNode};
use crate::types::{Confidence, CriterionType, Feature, Impurity, TreeConfig};
use std::collections::HashMap;

/// Error type for training failures.
#[derive(Debug, nexcore_error::Error)]
pub enum TrainError {
    /// No training data provided.
    #[error("no training data provided")]
    EmptyData,
    /// Feature vectors have inconsistent lengths.
    #[error("inconsistent feature vector lengths: expected {expected}, got {got} at row {row}")]
    InconsistentFeatures {
        expected: usize,
        got: usize,
        row: usize,
    },
    /// Labels count doesn't match data count.
    #[error("label count ({labels}) doesn't match data count ({data})")]
    LabelMismatch { labels: usize, data: usize },
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Train a classification decision tree on the given data.
///
/// # Errors
/// Returns `TrainError` if data is empty, inconsistent, or labels don't match.
pub fn fit(
    data: &[Vec<Feature>],
    labels: &[String],
    config: TreeConfig,
) -> Result<DecisionTree, TrainError> {
    validate_data(data, labels.len())?;

    let n_features = data[0].len();
    let n_classes = count_distinct_classes(labels);
    let criterion = make_criterion(config.criterion);
    let indices: Vec<usize> = (0..data.len()).collect();

    let ctx = ClassificationContext {
        data,
        labels,
        config: &config,
        criterion: criterion.as_ref(),
    };
    let root = ctx.build_node(&indices, 0);

    let mut tree = DecisionTree::new(config);
    tree.root = Some(root);
    tree.n_features = n_features;
    tree.n_classes = n_classes;
    tree.n_samples = data.len();
    tree.is_fitted = true;

    Ok(tree)
}

/// Train a regression decision tree.
///
/// # Errors
/// Returns `TrainError` if data is empty or inconsistent.
pub fn fit_regression(
    data: &[Vec<Feature>],
    targets: &[f64],
    config: TreeConfig,
) -> Result<DecisionTree, TrainError> {
    validate_data(data, targets.len())?;

    let n_features = data[0].len();
    let indices: Vec<usize> = (0..data.len()).collect();

    let ctx = RegressionContext {
        data,
        targets,
        config: &config,
    };
    let root = ctx.build_node(&indices, 0);

    let mut tree = DecisionTree::new(config);
    tree.root = Some(root);
    tree.n_features = n_features;
    tree.n_classes = 0;
    tree.n_samples = data.len();
    tree.is_fitted = true;

    Ok(tree)
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

fn validate_data(data: &[Vec<Feature>], label_count: usize) -> Result<(), TrainError> {
    if data.is_empty() {
        return Err(TrainError::EmptyData);
    }
    if label_count != data.len() {
        return Err(TrainError::LabelMismatch {
            labels: label_count,
            data: data.len(),
        });
    }
    let n_features = data[0].len();
    for (i, row) in data.iter().enumerate() {
        if row.len() != n_features {
            return Err(TrainError::InconsistentFeatures {
                expected: n_features,
                got: row.len(),
                row: i,
            });
        }
    }
    Ok(())
}

fn count_distinct_classes(labels: &[String]) -> usize {
    let mut sorted: Vec<&str> = labels.iter().map(String::as_str).collect();
    sorted.sort();
    sorted.dedup();
    sorted.len()
}

// ---------------------------------------------------------------------------
// Classification context — holds references, reduces parameter passing
// ---------------------------------------------------------------------------

struct ClassificationContext<'a> {
    data: &'a [Vec<Feature>],
    labels: &'a [String],
    config: &'a TreeConfig,
    criterion: &'a dyn SplitCriterion,
}

impl ClassificationContext<'_> {
    fn build_node(&self, indices: &[usize], depth: usize) -> TreeNode {
        let (class_counts, class_names, total) = class_distribution(self.labels, indices);
        let node_impurity = self.criterion.impurity(&class_counts, total);

        if self.should_stop(indices.len(), node_impurity, depth) {
            return make_classification_leaf(&class_counts, &class_names, total, node_impurity);
        }

        let n_features = self.data[0].len();
        let best = self.find_best_split(indices, n_features, node_impurity);

        let Some(split) = best else {
            return make_classification_leaf(&class_counts, &class_names, total, node_impurity);
        };

        let left = self.build_node(&split.left_indices, depth + 1);
        let right = self.build_node(&split.right_indices, depth + 1);

        TreeNode::Split {
            feature_index: split.feature_index,
            threshold: Feature::Continuous(split.threshold),
            left: Box::new(left),
            right: Box::new(right),
            impurity: node_impurity,
            samples: indices.len(),
            impurity_decrease: split.decrease,
        }
    }

    fn should_stop(&self, n_samples: usize, impurity: Impurity, depth: usize) -> bool {
        n_samples < self.config.min_samples_split
            || impurity.is_pure()
            || self.config.max_depth.is_some_and(|md| depth >= md)
    }

    fn find_best_split(
        &self,
        indices: &[usize],
        n_features: usize,
        parent_impurity: Impurity,
    ) -> Option<BestSplit> {
        let feature_range = match self.config.max_features {
            Some(max_f) if max_f < n_features => max_f,
            _ => n_features,
        };

        let mut best: Option<BestSplit> = None;
        for feat_idx in 0..feature_range {
            let candidate = self.evaluate_feature(indices, feat_idx, parent_impurity);
            best = pick_better(best, candidate);
        }
        best
    }

    fn evaluate_feature(
        &self,
        indices: &[usize],
        feat_idx: usize,
        parent_impurity: Impurity,
    ) -> Option<BestSplit> {
        let thresholds = extract_thresholds(self.data, indices, feat_idx);
        let n_samples = indices.len();
        let mut best: Option<BestSplit> = None;

        for threshold in thresholds {
            let split =
                self.evaluate_threshold(indices, feat_idx, threshold, parent_impurity, n_samples);
            best = pick_better(best, split);
        }
        best
    }

    fn evaluate_threshold(
        &self,
        indices: &[usize],
        feat_idx: usize,
        threshold: f64,
        parent_impurity: Impurity,
        n_samples: usize,
    ) -> Option<BestSplit> {
        let (left_idx, right_idx) = partition_indices(self.data, indices, feat_idx, threshold);

        if left_idx.len() < self.config.min_samples_leaf
            || right_idx.len() < self.config.min_samples_leaf
        {
            return None;
        }

        let decrease = self.compute_decrease(&left_idx, &right_idx, parent_impurity, n_samples);

        if decrease < self.config.min_impurity_decrease {
            return None;
        }

        Some(BestSplit {
            feature_index: feat_idx,
            threshold,
            left_indices: left_idx,
            right_indices: right_idx,
            decrease,
        })
    }

    fn compute_decrease(
        &self,
        left_idx: &[usize],
        right_idx: &[usize],
        parent_impurity: Impurity,
        n_samples: usize,
    ) -> f64 {
        let (left_counts, _, left_total) = class_distribution(self.labels, left_idx);
        let (right_counts, _, right_total) = class_distribution(self.labels, right_idx);

        let left_imp = self.criterion.impurity(&left_counts, left_total);
        let right_imp = self.criterion.impurity(&right_counts, right_total);

        let n_f = n_samples as f64;
        let raw_decrease = parent_impurity.0
            - (left_idx.len() as f64 / n_f) * left_imp.0
            - (right_idx.len() as f64 / n_f) * right_imp.0;

        if self.config.criterion == CriterionType::GainRatio {
            let si = split_info(left_idx.len(), right_idx.len());
            if si > f64::EPSILON {
                raw_decrease / si
            } else {
                raw_decrease
            }
        } else {
            raw_decrease
        }
    }
}

// ---------------------------------------------------------------------------
// Regression context
// ---------------------------------------------------------------------------

struct RegressionContext<'a> {
    data: &'a [Vec<Feature>],
    targets: &'a [f64],
    config: &'a TreeConfig,
}

impl RegressionContext<'_> {
    fn build_node(&self, indices: &[usize], depth: usize) -> TreeNode {
        let target_values: Vec<f64> = indices.iter().map(|&i| self.targets[i]).collect();
        let node_impurity = Mse::impurity_from_values(&target_values);
        let mean = compute_mean(&target_values);

        let should_stop = indices.len() < self.config.min_samples_split
            || node_impurity.is_pure()
            || self.config.max_depth.is_some_and(|md| depth >= md);

        if should_stop {
            return make_regression_leaf(mean, node_impurity, indices.len());
        }

        let n_features = self.data[0].len();
        let best = self.find_best_split(indices, n_features, node_impurity);

        let Some(split) = best else {
            return make_regression_leaf(mean, node_impurity, indices.len());
        };

        let left = self.build_node(&split.left_indices, depth + 1);
        let right = self.build_node(&split.right_indices, depth + 1);

        TreeNode::Split {
            feature_index: split.feature_index,
            threshold: Feature::Continuous(split.threshold),
            left: Box::new(left),
            right: Box::new(right),
            impurity: node_impurity,
            samples: indices.len(),
            impurity_decrease: split.decrease,
        }
    }

    fn find_best_split(
        &self,
        indices: &[usize],
        n_features: usize,
        parent_impurity: Impurity,
    ) -> Option<BestSplit> {
        let mut best: Option<BestSplit> = None;
        for feat_idx in 0..n_features {
            let candidate = self.evaluate_feature(indices, feat_idx, parent_impurity);
            best = pick_better(best, candidate);
        }
        best
    }

    fn evaluate_feature(
        &self,
        indices: &[usize],
        feat_idx: usize,
        parent_impurity: Impurity,
    ) -> Option<BestSplit> {
        let thresholds = extract_thresholds(self.data, indices, feat_idx);
        let n_samples = indices.len();
        let mut best: Option<BestSplit> = None;

        for threshold in thresholds {
            let split =
                self.evaluate_threshold(indices, feat_idx, threshold, parent_impurity, n_samples);
            best = pick_better(best, split);
        }
        best
    }

    fn evaluate_threshold(
        &self,
        indices: &[usize],
        feat_idx: usize,
        threshold: f64,
        parent_impurity: Impurity,
        n_samples: usize,
    ) -> Option<BestSplit> {
        let (left_idx, right_idx) = partition_indices(self.data, indices, feat_idx, threshold);

        if left_idx.len() < self.config.min_samples_leaf
            || right_idx.len() < self.config.min_samples_leaf
        {
            return None;
        }

        let left_vals: Vec<f64> = left_idx.iter().map(|&i| self.targets[i]).collect();
        let right_vals: Vec<f64> = right_idx.iter().map(|&i| self.targets[i]).collect();

        let left_imp = Mse::impurity_from_values(&left_vals);
        let right_imp = Mse::impurity_from_values(&right_vals);

        let n_f = n_samples as f64;
        let decrease = parent_impurity.0
            - (left_idx.len() as f64 / n_f) * left_imp.0
            - (right_idx.len() as f64 / n_f) * right_imp.0;

        if decrease < self.config.min_impurity_decrease {
            return None;
        }

        Some(BestSplit {
            feature_index: feat_idx,
            threshold,
            left_indices: left_idx,
            right_indices: right_idx,
            decrease,
        })
    }
}

// ---------------------------------------------------------------------------
// Shared helpers (T1: σ sequence + μ mapping)
// ---------------------------------------------------------------------------

/// Temporary struct to hold the best split candidate during search.
struct BestSplit {
    feature_index: usize,
    threshold: f64,
    left_indices: Vec<usize>,
    right_indices: Vec<usize>,
    decrease: f64,
}

/// Pick the better of two optional splits by decrease.
fn pick_better(current: Option<BestSplit>, candidate: Option<BestSplit>) -> Option<BestSplit> {
    match (current, candidate) {
        (None, c) => c,
        (c, None) => c,
        (Some(a), Some(b)) => {
            if b.decrease > a.decrease {
                Some(b)
            } else {
                Some(a)
            }
        }
    }
}

/// Compute class distribution from labels at given indices.
fn class_distribution(labels: &[String], indices: &[usize]) -> (Vec<usize>, Vec<String>, usize) {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for &i in indices {
        *counts.entry(labels[i].as_str()).or_insert(0) += 1;
    }
    let mut class_names: Vec<String> = counts.keys().map(|k| k.to_string()).collect();
    class_names.sort();
    let class_counts: Vec<usize> = class_names
        .iter()
        .map(|name| counts.get(name.as_str()).copied().unwrap_or(0))
        .collect();
    (class_counts, class_names, indices.len())
}

/// Extract sorted unique thresholds (midpoints) for a feature.
fn extract_thresholds(data: &[Vec<Feature>], indices: &[usize], feat_idx: usize) -> Vec<f64> {
    let mut vals: Vec<f64> = indices
        .iter()
        .filter_map(|&i| data[i][feat_idx].as_continuous())
        .collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut thresholds = Vec::new();
    if vals.is_empty() {
        return thresholds;
    }
    let mut prev = vals[0];
    for &val in vals.iter().skip(1) {
        if (val - prev).abs() > f64::EPSILON {
            thresholds.push((prev + val) / 2.0);
        }
        prev = val;
    }
    thresholds
}

/// Partition indices by continuous threshold. Left: ≤ threshold, Right: > threshold.
fn partition_indices(
    data: &[Vec<Feature>],
    indices: &[usize],
    feature_index: usize,
    threshold: f64,
) -> (Vec<usize>, Vec<usize>) {
    let mut left = Vec::new();
    let mut right = Vec::new();
    for &i in indices {
        let goes_left = data[i][feature_index]
            .as_continuous()
            .is_some_and(|v| v <= threshold);
        if goes_left {
            left.push(i);
        } else {
            right.push(i);
        }
    }
    (left, right)
}

/// Compute mean of a slice.
fn compute_mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Create a classification leaf node.
fn make_classification_leaf(
    class_counts: &[usize],
    class_names: &[String],
    total: usize,
    impurity: Impurity,
) -> TreeNode {
    if total == 0 {
        return TreeNode::Leaf {
            prediction: String::new(),
            confidence: Confidence::NONE,
            distribution: Vec::new(),
            samples: 0,
            impurity: Impurity::PURE,
        };
    }

    let total_f = total as f64;
    let distribution: Vec<(String, f64)> = class_names
        .iter()
        .zip(class_counts.iter())
        .map(|(name, &count)| (name.clone(), count as f64 / total_f))
        .collect();

    let (best_idx, best_count) = class_counts
        .iter()
        .enumerate()
        .max_by_key(|&(_, c)| *c)
        .map(|(i, c)| (i, *c))
        .unwrap_or((0, 0));

    let prediction = class_names.get(best_idx).cloned().unwrap_or_default();
    let confidence = Confidence::new(best_count as f64 / total_f);

    TreeNode::Leaf {
        prediction,
        confidence,
        distribution,
        samples: total,
        impurity,
    }
}

/// Create a regression leaf node.
fn make_regression_leaf(mean: f64, impurity: Impurity, samples: usize) -> TreeNode {
    let conf = if impurity.is_pure() {
        1.0
    } else {
        1.0 / (1.0 + impurity.0)
    };
    TreeNode::Leaf {
        prediction: format!("{mean:.6}"),
        confidence: Confidence::new(conf),
        distribution: vec![(format!("{mean:.6}"), 1.0)],
        samples,
        impurity,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::CriterionType;

    fn make_xor_data() -> (Vec<Vec<Feature>>, Vec<String>) {
        let data = vec![
            vec![Feature::Continuous(0.0), Feature::Continuous(0.0)],
            vec![Feature::Continuous(0.0), Feature::Continuous(1.0)],
            vec![Feature::Continuous(1.0), Feature::Continuous(0.0)],
            vec![Feature::Continuous(1.0), Feature::Continuous(1.0)],
        ];
        let labels = vec!["0".into(), "1".into(), "1".into(), "0".into()];
        (data, labels)
    }

    fn make_simple_data() -> (Vec<Vec<Feature>>, Vec<String>) {
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
        (data, labels)
    }

    #[test]
    fn fit_empty_data_returns_error() {
        let result = fit(&[], &[], TreeConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn fit_label_mismatch_returns_error() {
        let data = vec![vec![Feature::Continuous(1.0)]];
        let labels: Vec<String> = Vec::new();
        assert!(fit(&data, &labels, TreeConfig::default()).is_err());
    }

    #[test]
    fn fit_simple_linearly_separable() {
        let (data, labels) = make_simple_data();
        let tree = fit(&data, &labels, TreeConfig::default());
        assert!(tree.is_ok());
        let tree = tree.ok();
        assert!(tree.is_some());
        let tree = tree.unwrap_or_else(|| DecisionTree::new(TreeConfig::default()));
        assert!(tree.is_fitted());
    }

    #[test]
    fn fit_xor_requires_depth() {
        let (data, labels) = make_xor_data();
        let tree = fit(&data, &labels, TreeConfig::default());
        assert!(tree.is_ok());
    }

    #[test]
    fn fit_with_max_depth() {
        let (data, labels) = make_xor_data();
        let config = TreeConfig {
            max_depth: Some(1),
            ..TreeConfig::default()
        };
        let result = fit(&data, &labels, config);
        assert!(result.is_ok());
    }

    #[test]
    fn fit_with_entropy() {
        let (data, labels) = make_simple_data();
        let config = TreeConfig {
            criterion: CriterionType::Entropy,
            ..TreeConfig::default()
        };
        assert!(fit(&data, &labels, config).is_ok());
    }

    #[test]
    fn fit_regression_simple() {
        let data = vec![
            vec![Feature::Continuous(1.0)],
            vec![Feature::Continuous(2.0)],
            vec![Feature::Continuous(3.0)],
            vec![Feature::Continuous(4.0)],
        ];
        let targets = vec![1.0, 2.0, 3.0, 4.0];
        let config = TreeConfig {
            criterion: CriterionType::Mse,
            ..TreeConfig::default()
        };
        let result = fit_regression(&data, &targets, config);
        assert!(result.is_ok());
    }

    #[test]
    fn extract_thresholds_basic() {
        let data = vec![
            vec![Feature::Continuous(1.0)],
            vec![Feature::Continuous(2.0)],
            vec![Feature::Continuous(3.0)],
        ];
        let thresholds = extract_thresholds(&data, &[0, 1, 2], 0);
        assert_eq!(thresholds.len(), 2);
        assert!((thresholds[0] - 1.5).abs() < f64::EPSILON);
        assert!((thresholds[1] - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn partition_indices_basic() {
        let data = vec![
            vec![Feature::Continuous(1.0)],
            vec![Feature::Continuous(2.0)],
            vec![Feature::Continuous(3.0)],
        ];
        let (left, right) = partition_indices(&data, &[0, 1, 2], 0, 1.5);
        assert_eq!(left, vec![0]);
        assert_eq!(right, vec![1, 2]);
    }
}
