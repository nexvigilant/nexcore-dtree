//! Core types for the decision tree engine.
//!
//! Grounded to the Primitive Codex:
//! - T2-P newtypes: `Impurity`, `Confidence`
//! - T2-C composed: `Feature`, `SplitPoint`, `TreeConfig`, `PredictionResult`

use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// T2-P Newtypes
// ---------------------------------------------------------------------------

/// Tier: T2-P (newtype over T1 f64)
///
/// Impurity measure for a node. Range depends on criterion:
/// - Gini: [0.0, 1.0 - 1/C] where C = number of classes
/// - Entropy: [0.0, log2(C)]
/// - MSE: [0.0, ∞)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Impurity(pub f64);

impl Impurity {
    /// A pure node (all samples belong to one class).
    pub const PURE: Self = Self(0.0);

    /// Returns `true` if the node is pure (impurity ≈ 0).
    #[must_use]
    pub fn is_pure(self) -> bool {
        self.0.abs() < f64::EPSILON
    }
}

impl fmt::Display for Impurity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.6}", self.0)
    }
}

/// Tier: T2-P (N + ∂ — bounded probability)
///
/// Canonical confidence score in [0.0, 1.0].
/// Re-exported from `nexcore-constants` to eliminate F2 equivocation.
pub use nexcore_constants::Confidence;

// ---------------------------------------------------------------------------
// T2-C Composed Types
// ---------------------------------------------------------------------------

/// Tier: T2-C (composed)
///
/// A single feature value. Supports continuous, categorical, and missing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Feature {
    /// Numeric feature (f64).
    Continuous(f64),
    /// Categorical feature (string label).
    Categorical(String),
    /// Missing / unknown value.
    Missing,
}

impl Feature {
    /// Extract continuous value, returning `None` for non-continuous.
    #[must_use]
    pub fn as_continuous(&self) -> Option<f64> {
        match self {
            Self::Continuous(v) => Some(*v),
            _ => None,
        }
    }

    /// Extract categorical value, returning `None` for non-categorical.
    #[must_use]
    pub fn as_categorical(&self) -> Option<&str> {
        match self {
            Self::Categorical(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Returns `true` if this feature is missing.
    #[must_use]
    pub fn is_missing(&self) -> bool {
        matches!(self, Self::Missing)
    }
}

impl fmt::Display for Feature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Continuous(v) => write!(f, "{v:.4}"),
            Self::Categorical(s) => write!(f, "{s}"),
            Self::Missing => write!(f, "?"),
        }
    }
}

/// Tier: T2-C (composed)
///
/// Describes the best split found at a node during training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitPoint {
    /// Index of the feature used for this split.
    pub feature_index: usize,
    /// Threshold value for the split.
    pub threshold: Feature,
    /// Impurity decrease achieved by this split.
    pub impurity_decrease: Impurity,
    /// Number of samples going to the left child.
    pub samples_left: usize,
    /// Number of samples going to the right child.
    pub samples_right: usize,
}

/// Which splitting criterion to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CriterionType {
    /// Gini impurity: 1 - Σ(p_i²). CART default.
    Gini,
    /// Shannon entropy: -Σ(p_i * log2(p_i)). ID3/C4.5.
    Entropy,
    /// Gain ratio: IG / SplitInfo. C4.5 correction for cardinality bias.
    GainRatio,
    /// Mean squared error. For regression trees.
    Mse,
}

impl Default for CriterionType {
    fn default() -> Self {
        Self::Gini
    }
}

/// Tier: T2-C (composed)
///
/// Configuration for training a decision tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeConfig {
    /// Pre-prune: maximum tree depth. `None` = unlimited.
    pub max_depth: Option<usize>,
    /// Pre-prune: minimum samples required to attempt a split (default: 2).
    pub min_samples_split: usize,
    /// Pre-prune: minimum samples required in each leaf (default: 1).
    pub min_samples_leaf: usize,
    /// Pre-prune: minimum impurity decrease to accept a split (default: 0.0).
    pub min_impurity_decrease: f64,
    /// Feature subsampling: consider at most this many features per split.
    /// `None` = use all features. (Useful for random forest extension.)
    pub max_features: Option<usize>,
    /// Which splitting criterion to use.
    pub criterion: CriterionType,
}

impl Default for TreeConfig {
    fn default() -> Self {
        Self {
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            min_impurity_decrease: 0.0,
            max_features: None,
            criterion: CriterionType::default(),
        }
    }
}

/// Direction taken at a split node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    /// Left child: ≤ threshold (continuous) or ∈ subset (categorical).
    Left,
    /// Right child: > threshold (continuous) or ∉ subset (categorical).
    Right,
}

impl fmt::Display for Direction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Left => write!(f, "left"),
            Self::Right => write!(f, "right"),
        }
    }
}

/// Describes one step in the prediction path through the tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitDescription {
    /// Index of the feature evaluated.
    pub feature_index: usize,
    /// Human-readable feature name, if available.
    pub feature_name: Option<String>,
    /// String representation of the threshold.
    pub threshold: String,
    /// Which direction was taken.
    pub direction: Direction,
}

impl fmt::Display for SplitDescription {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let fallback = format!("feature[{}]", self.feature_index);
        let name = self.feature_name.as_deref().unwrap_or(&fallback);
        let op = match self.direction {
            Direction::Left => "<=",
            Direction::Right => ">",
        };
        write!(f, "{name} {op} {}", self.threshold)
    }
}

/// Tier: T2-C (composed)
///
/// Result of a prediction, including confidence and explainability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    /// The predicted class label (as string).
    pub prediction: String,
    /// Confidence of the prediction.
    pub confidence: Confidence,
    /// Class distribution at the leaf: (class_label, probability).
    pub class_distribution: Vec<(String, f64)>,
    /// Number of training samples that reached this leaf.
    pub leaf_samples: usize,
    /// Depth of the leaf in the tree.
    pub depth: usize,
    /// Explainability: the rule path taken from root to leaf.
    pub path: Vec<SplitDescription>,
}

/// Tier: T2-C (composed)
///
/// Result of a regression prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionResult {
    /// The predicted numeric value.
    pub prediction: f64,
    /// Variance of training samples at the leaf (as confidence proxy).
    pub variance: f64,
    /// Number of training samples that reached this leaf.
    pub leaf_samples: usize,
    /// Depth of the leaf in the tree.
    pub depth: usize,
    /// Explainability: the rule path taken from root to leaf.
    pub path: Vec<SplitDescription>,
}

/// Feature importance entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureImportance {
    /// Feature index.
    pub index: usize,
    /// Feature name, if available.
    pub name: Option<String>,
    /// Importance score (sum of weighted impurity decreases).
    pub importance: f64,
}

/// Tree statistics summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeStats {
    /// Maximum depth of the tree.
    pub depth: usize,
    /// Number of leaf nodes.
    pub n_leaves: usize,
    /// Number of internal (split) nodes.
    pub n_splits: usize,
    /// Total number of nodes.
    pub n_nodes: usize,
    /// Number of features the tree was trained on.
    pub n_features: usize,
    /// Number of distinct classes (0 for regression).
    pub n_classes: usize,
    /// Number of training samples.
    pub n_samples: usize,
    /// Criterion used for training.
    pub criterion: CriterionType,
}
