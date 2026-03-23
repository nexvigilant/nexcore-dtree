//! Tree node structure for the decision tree.
//!
//! ## Primitive Foundation
//! - T1: Recursion (ρ) — `TreeNode` is a recursive enum (Split → children → Leaf)
//! - T1: State (ς) — each node holds impurity, sample count, distribution
//! - T1: Exists (∃) — leaf prediction existence

use crate::types::{Confidence, CriterionType, Feature, Impurity, TreeConfig, TreeStats};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// TreeNode — recursive tree structure
// ---------------------------------------------------------------------------

/// Tier: T3 (full domain type)
///
/// A node in the decision tree. Either a split (internal) or leaf (terminal).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TreeNode {
    /// Internal split node.
    Split {
        /// Index of the feature used for splitting.
        feature_index: usize,
        /// Threshold value. For continuous: samples ≤ threshold go left.
        /// For categorical: samples matching threshold go left.
        threshold: Feature,
        /// Left subtree (≤ threshold or ∈ subset).
        left: Box<TreeNode>,
        /// Right subtree (> threshold or ∉ subset).
        right: Box<TreeNode>,
        /// Impurity of this node before splitting.
        impurity: Impurity,
        /// Number of training samples at this node.
        samples: usize,
        /// Weighted impurity decrease from this split (for feature importance).
        impurity_decrease: f64,
    },
    /// Terminal leaf node.
    Leaf {
        /// Predicted class label (majority class for classification).
        prediction: String,
        /// Confidence = proportion of majority class.
        confidence: Confidence,
        /// Class distribution: (class_label, proportion).
        distribution: Vec<(String, f64)>,
        /// Number of training samples at this leaf.
        samples: usize,
        /// Impurity of this leaf node.
        impurity: Impurity,
    },
}

impl TreeNode {
    /// Returns `true` if this is a leaf node.
    #[must_use]
    pub fn is_leaf(&self) -> bool {
        matches!(self, Self::Leaf { .. })
    }

    /// Returns `true` if this is a split node.
    #[must_use]
    pub fn is_split(&self) -> bool {
        matches!(self, Self::Split { .. })
    }

    /// Get the number of training samples at this node.
    #[must_use]
    pub fn samples(&self) -> usize {
        match self {
            Self::Split { samples, .. } | Self::Leaf { samples, .. } => *samples,
        }
    }

    /// Get the impurity of this node.
    #[must_use]
    pub fn impurity(&self) -> Impurity {
        match self {
            Self::Split { impurity, .. } | Self::Leaf { impurity, .. } => *impurity,
        }
    }

    /// Compute the depth of the tree rooted at this node.
    #[must_use]
    pub fn depth(&self) -> usize {
        match self {
            Self::Leaf { .. } => 0,
            Self::Split { left, right, .. } => 1 + left.depth().max(right.depth()),
        }
    }

    /// Count the number of leaf nodes in the subtree.
    #[must_use]
    pub fn n_leaves(&self) -> usize {
        match self {
            Self::Leaf { .. } => 1,
            Self::Split { left, right, .. } => left.n_leaves() + right.n_leaves(),
        }
    }

    /// Count all nodes (leaves + splits) in the subtree.
    #[must_use]
    pub fn n_nodes(&self) -> usize {
        match self {
            Self::Leaf { .. } => 1,
            Self::Split { left, right, .. } => 1 + left.n_nodes() + right.n_nodes(),
        }
    }

    /// Compute the total misclassification cost of this node if it were a leaf.
    /// For classification: 1.0 - max(p_i) = 1.0 - confidence.
    /// Weighted by number of samples.
    #[must_use]
    pub fn leaf_cost(&self) -> f64 {
        let (confidence, samples) = match self {
            Self::Leaf {
                confidence,
                samples,
                ..
            } => (confidence.value(), *samples),
            Self::Split { samples, .. } => {
                // If this split were collapsed to a leaf, we'd need the distribution.
                // Use impurity as proxy.
                let imp = self.impurity();
                (1.0 - imp.0, *samples)
            }
        };
        (1.0 - confidence) * samples as f64
    }

    /// Compute the total subtree misclassification cost.
    /// Sum of leaf costs across all leaves in the subtree.
    #[must_use]
    pub fn subtree_cost(&self) -> f64 {
        match self {
            Self::Leaf {
                confidence,
                samples,
                ..
            } => (1.0 - confidence.value()) * *samples as f64,
            Self::Split { left, right, .. } => left.subtree_cost() + right.subtree_cost(),
        }
    }
}

// ---------------------------------------------------------------------------
// DecisionTree — the trained model
// ---------------------------------------------------------------------------

/// Tier: T3 (full domain type)
///
/// A trained decision tree model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTree {
    /// Root node of the tree. `None` before fitting.
    pub(crate) root: Option<TreeNode>,
    /// Training configuration.
    pub(crate) config: TreeConfig,
    /// Feature names (optional, for explainability).
    pub(crate) feature_names: Vec<String>,
    /// Number of features the tree was trained on.
    pub(crate) n_features: usize,
    /// Number of distinct classes (0 for regression).
    pub(crate) n_classes: usize,
    /// Total training samples.
    pub(crate) n_samples: usize,
    /// Whether the tree has been fitted.
    pub(crate) is_fitted: bool,
}

impl DecisionTree {
    /// Create a new unfitted decision tree with the given config.
    #[must_use]
    pub fn new(config: TreeConfig) -> Self {
        Self {
            root: None,
            config,
            feature_names: Vec::new(),
            n_features: 0,
            n_classes: 0,
            n_samples: 0,
            is_fitted: false,
        }
    }

    /// Returns `true` if the tree has been fitted.
    #[must_use]
    pub fn is_fitted(&self) -> bool {
        self.is_fitted
    }

    /// Get the tree configuration.
    #[must_use]
    pub fn config(&self) -> &TreeConfig {
        &self.config
    }

    /// Get feature names.
    #[must_use]
    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    /// Set feature names for explainability.
    pub fn set_feature_names(&mut self, names: Vec<String>) {
        self.feature_names = names;
    }

    /// Get the root node, if fitted.
    #[must_use]
    pub fn root(&self) -> Option<&TreeNode> {
        self.root.as_ref()
    }

    /// Get a mutable reference to the root node, if fitted.
    pub fn root_mut(&mut self) -> Option<&mut TreeNode> {
        self.root.as_mut()
    }

    /// Compute tree statistics.
    #[must_use]
    pub fn stats(&self) -> Option<TreeStats> {
        let root = self.root.as_ref()?;
        let n_leaves = root.n_leaves();
        let n_nodes = root.n_nodes();
        Some(TreeStats {
            depth: root.depth(),
            n_leaves,
            n_splits: n_nodes - n_leaves,
            n_nodes,
            n_features: self.n_features,
            n_classes: self.n_classes,
            n_samples: self.n_samples,
            criterion: self.config.criterion,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_leaf(prediction: &str, confidence: f64, samples: usize) -> TreeNode {
        TreeNode::Leaf {
            prediction: prediction.to_string(),
            confidence: Confidence::new(confidence),
            distribution: vec![(prediction.to_string(), confidence)],
            samples,
            impurity: Impurity(1.0 - confidence),
        }
    }

    fn make_split(left: TreeNode, right: TreeNode, samples: usize) -> TreeNode {
        TreeNode::Split {
            feature_index: 0,
            threshold: Feature::Continuous(0.5),
            left: Box::new(left),
            right: Box::new(right),
            impurity: Impurity(0.5),
            samples,
            impurity_decrease: 0.1,
        }
    }

    #[test]
    fn leaf_is_leaf() {
        let leaf = make_leaf("A", 1.0, 10);
        assert!(leaf.is_leaf());
        assert!(!leaf.is_split());
    }

    #[test]
    fn split_is_split() {
        let node = make_split(make_leaf("A", 1.0, 5), make_leaf("B", 1.0, 5), 10);
        assert!(node.is_split());
        assert!(!node.is_leaf());
    }

    #[test]
    fn leaf_depth_is_zero() {
        let leaf = make_leaf("A", 1.0, 10);
        assert_eq!(leaf.depth(), 0);
    }

    #[test]
    fn one_level_depth() {
        let node = make_split(make_leaf("A", 1.0, 5), make_leaf("B", 1.0, 5), 10);
        assert_eq!(node.depth(), 1);
    }

    #[test]
    fn two_level_depth() {
        let inner = make_split(make_leaf("A", 1.0, 3), make_leaf("B", 1.0, 2), 5);
        let node = make_split(inner, make_leaf("C", 1.0, 5), 10);
        assert_eq!(node.depth(), 2);
    }

    #[test]
    fn leaf_count() {
        let node = make_split(
            make_split(make_leaf("A", 1.0, 3), make_leaf("B", 1.0, 2), 5),
            make_leaf("C", 1.0, 5),
            10,
        );
        assert_eq!(node.n_leaves(), 3);
    }

    #[test]
    fn node_count() {
        let node = make_split(
            make_split(make_leaf("A", 1.0, 3), make_leaf("B", 1.0, 2), 5),
            make_leaf("C", 1.0, 5),
            10,
        );
        // 2 splits + 3 leaves = 5 nodes
        assert_eq!(node.n_nodes(), 5);
    }

    #[test]
    fn decision_tree_new_is_unfitted() {
        let tree = DecisionTree::new(TreeConfig::default());
        assert!(!tree.is_fitted());
        assert!(tree.root().is_none());
        assert!(tree.stats().is_none());
    }
}
