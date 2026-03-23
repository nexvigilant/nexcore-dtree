//! # NexVigilant Core — dtree
//!
//! CART-inspired decision tree engine with pluggable splitting criteria.
//!
//! ## Primitive Foundation
//!
//! | Primitive | Manifestation |
//! |-----------|---------------|
//! | T1: Recursion (ρ) | Binary tree traversal (split → left/right → leaf) |
//! | T1: Mapping (μ) | Feature → split threshold → child branch |
//! | T1: Sequence (σ) | Feature evaluation order, pruning sequence |
//! | T1: State (ς) | Node impurity, sample distribution, learned splits |
//! | T1: Exists (∃) | Leaf prediction existence check |
//!
//! ## Quick Start
//!
//! ```rust
//! use nexcore_dtree::prelude::*;
//!
//! let config = TreeConfig::default();
//! let tree = DecisionTree::new(config);
//! // Use train::fit() to train on data, then predict::predict() for inference.
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

pub mod criterion;
pub mod grounding;
pub mod importance;
pub mod node;
pub mod predict;
pub mod prune;
pub mod serialize;
pub mod spatial_bridge;
pub mod train;
pub mod types;

/// Convenience prelude for common imports.
pub mod prelude {
    pub use crate::criterion::{Entropy, Gini, Mse, SplitCriterion, make_criterion};
    pub use crate::importance::feature_importance;
    pub use crate::node::{DecisionTree, TreeNode};
    pub use crate::predict::{predict, predict_batch, predict_regression};
    pub use crate::prune::{cost_complexity_prune, pruning_path};
    pub use crate::serialize::{from_json, to_json, to_rules, to_summary};
    pub use crate::train::{fit, fit_regression};
    pub use crate::types::{
        Confidence, CriterionType, Feature, FeatureImportance, Impurity, PredictionResult,
        RegressionResult, SplitDescription, SplitPoint, TreeConfig, TreeStats,
    };
}
