//! # GroundsTo implementations for nexcore-dtree types
//!
//! Connects decision tree types to the Lex Primitiva type system.
//!
//! ## ρ (Recursion) Focus
//!
//! Decision trees ARE recursive structures: split → left/right → leaf.
//! The crate's grammar is Type-2 (context-free): ρ provides pushdown
//! traversal, κ drives split comparisons, μ maps features to branches.

use nexcore_lex_primitiva::grounding::GroundsTo;
use nexcore_lex_primitiva::primitiva::{LexPrimitiva, PrimitiveComposition};

use crate::node::TreeNode;
use crate::types::{
    CriterionType, Direction, Feature, FeatureImportance, Impurity, PredictionResult,
    RegressionResult, SplitDescription, SplitPoint, TreeConfig, TreeStats,
};

// ---------------------------------------------------------------------------
// T2-P newtypes
// ---------------------------------------------------------------------------

/// Impurity: T1 (N), dominant N
///
/// Pure numeric measurement of node impurity.
/// Single primitive — irreducible quantity.
impl GroundsTo for Impurity {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::Quantity, // N — numeric impurity value
        ])
        .with_dominant(LexPrimitiva::Quantity, 0.95)
    }
}

// Confidence: GroundsTo impl is in nexcore-constants::grounding (canonical source).
// Removed local impl to eliminate F2 equivocation — see vocabulary::CONFIDENCE.

// ---------------------------------------------------------------------------
// Enum types — Σ dominant
// ---------------------------------------------------------------------------

/// Feature: T2-P (Σ · N · ∅), dominant Σ
///
/// Three-variant feature value: Continuous | Categorical | Missing.
/// Sum-dominant: the type IS an alternation over data representations.
impl GroundsTo for Feature {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::Sum,      // Σ — variant alternation
            LexPrimitiva::Quantity, // N — continuous numeric value
            LexPrimitiva::Void,     // ∅ — Missing variant
        ])
        .with_dominant(LexPrimitiva::Sum, 0.85)
    }
}

/// CriterionType: T1 (Σ), dominant Σ
///
/// Splitting criterion: Gini | Entropy | GainRatio | Mse.
/// Pure sum type — finite set of algorithm variants.
impl GroundsTo for CriterionType {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::Sum, // Σ — four-variant alternation
        ])
        .with_dominant(LexPrimitiva::Sum, 0.95)
    }
}

/// Direction: T2-P (Σ · κ), dominant κ
///
/// Left/Right traversal direction at a split node.
/// Comparison-dominant: direction is the outcome of a threshold comparison.
impl GroundsTo for Direction {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::Sum,        // Σ — binary alternation
            LexPrimitiva::Comparison, // κ — ≤ or > threshold
        ])
        .with_dominant(LexPrimitiva::Comparison, 0.85)
    }
}

// ---------------------------------------------------------------------------
// Split types — κ dominant
// ---------------------------------------------------------------------------

/// SplitPoint: T2-C (κ · N · ∂ · μ), dominant κ
///
/// The best split found during training at a node.
/// Comparison-dominant: a split IS a threshold comparison on a feature.
impl GroundsTo for SplitPoint {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::Comparison, // κ — threshold comparison
            LexPrimitiva::Quantity,   // N — impurity decrease, sample counts
            LexPrimitiva::Boundary,   // ∂ — threshold boundary
            LexPrimitiva::Mapping,    // μ — feature → split mapping
        ])
        .with_dominant(LexPrimitiva::Comparison, 0.85)
    }
}

/// SplitDescription: T2-P (κ · μ · λ), dominant κ
///
/// Human-readable description of one step in a prediction path.
/// Comparison-dominant: describes a threshold comparison.
impl GroundsTo for SplitDescription {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::Comparison, // κ — threshold comparison
            LexPrimitiva::Mapping,    // μ — feature → direction
            LexPrimitiva::Location,   // λ — feature identity
        ])
        .with_dominant(LexPrimitiva::Comparison, 0.85)
    }
}

// ---------------------------------------------------------------------------
// Config types — ∂ dominant
// ---------------------------------------------------------------------------

/// TreeConfig: T2-C (∂ · N · μ · Σ), dominant ∂
///
/// Training configuration with pre-pruning boundaries.
/// Boundary-dominant: config exists to set limits on tree growth.
impl GroundsTo for TreeConfig {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::Boundary, // ∂ — max_depth, min_samples limits
            LexPrimitiva::Quantity, // N — numeric parameters
            LexPrimitiva::Mapping,  // μ — config → behavior
            LexPrimitiva::Sum,      // Σ — criterion type selection
        ])
        .with_dominant(LexPrimitiva::Boundary, 0.85)
    }
}

// ---------------------------------------------------------------------------
// Result types — multi-primitive
// ---------------------------------------------------------------------------

/// PredictionResult: T3 (κ · N · σ · ρ · ∃ · μ), dominant κ
///
/// Classification prediction with confidence, distribution, and explainability.
/// Comparison-dominant: the prediction IS the result of recursive comparisons.
impl GroundsTo for PredictionResult {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::Comparison, // κ — split-based classification
            LexPrimitiva::Quantity,   // N — confidence, sample counts
            LexPrimitiva::Sequence,   // σ — class distribution, path
            LexPrimitiva::Recursion,  // ρ — recursive path through tree
            LexPrimitiva::Existence,  // ∃ — prediction exists
            LexPrimitiva::Mapping,    // μ — features → prediction
        ])
        .with_dominant(LexPrimitiva::Comparison, 0.80)
    }
}

/// RegressionResult: T2-C (N · σ · ρ · ∃), dominant N
///
/// Regression prediction with variance and path.
/// Quantity-dominant: regression produces a numeric prediction.
impl GroundsTo for RegressionResult {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::Quantity,  // N — predicted value, variance
            LexPrimitiva::Sequence,  // σ — decision path
            LexPrimitiva::Recursion, // ρ — recursive tree traversal
            LexPrimitiva::Existence, // ∃ — prediction exists
        ])
        .with_dominant(LexPrimitiva::Quantity, 0.85)
    }
}

/// FeatureImportance: T2-P (N · κ · λ), dominant N
///
/// Feature importance score with identity.
/// Quantity-dominant: importance IS a numeric score.
impl GroundsTo for FeatureImportance {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::Quantity,   // N — importance score
            LexPrimitiva::Comparison, // κ — importance ranking
            LexPrimitiva::Location,   // λ — feature identity
        ])
        .with_dominant(LexPrimitiva::Quantity, 0.85)
    }
}

/// TreeStats: T2-C (N · Σ · κ · ρ), dominant N
///
/// Summary statistics of a trained tree.
/// Quantity-dominant: all fields are numeric counts.
impl GroundsTo for TreeStats {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::Quantity,   // N — depth, counts
            LexPrimitiva::Sum,        // Σ — aggregated totals
            LexPrimitiva::Comparison, // κ — criterion type
            LexPrimitiva::Recursion,  // ρ — tree structure stats
        ])
        .with_dominant(LexPrimitiva::Quantity, 0.85)
    }
}

// ---------------------------------------------------------------------------
// Tree node — ρ dominant (T3)
// ---------------------------------------------------------------------------

/// TreeNode: T3 (ρ · κ · ς · N · μ · ∃), dominant ρ
///
/// Recursive tree node: Split(feature, threshold, left, right) | Leaf(prediction).
/// Recursion-dominant: the type IS recursive — Box<TreeNode> in Split variant.
/// This is the canonical ρ type in the decision tree domain.
impl GroundsTo for TreeNode {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::Recursion,  // ρ — Box<TreeNode> self-reference
            LexPrimitiva::Comparison, // κ — threshold splits
            LexPrimitiva::State,      // ς — impurity, sample distribution
            LexPrimitiva::Quantity,   // N — impurity values, sample counts
            LexPrimitiva::Mapping,    // μ — feature → branch mapping
            LexPrimitiva::Existence,  // ∃ — leaf prediction existence
        ])
        .with_dominant(LexPrimitiva::Recursion, 0.90)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Confidence;
    use nexcore_lex_primitiva::tier::Tier;

    #[test]
    fn impurity_is_t1() {
        assert_eq!(Impurity::tier(), Tier::T1Universal);
        assert_eq!(
            Impurity::primitive_composition().dominant,
            Some(LexPrimitiva::Quantity)
        );
    }

    #[test]
    fn confidence_is_t2p() {
        assert_eq!(Confidence::tier(), Tier::T2Primitive);
        // Canonical: N + κ (Quantity + Comparison)
        assert!(
            Confidence::primitive_composition()
                .primitives
                .contains(&LexPrimitiva::Comparison)
        );
    }

    #[test]
    fn feature_is_sum_dominant() {
        let comp = Feature::primitive_composition();
        assert_eq!(comp.dominant, Some(LexPrimitiva::Sum));
        assert!(comp.primitives.contains(&LexPrimitiva::Void));
    }

    #[test]
    fn criterion_type_is_t1() {
        assert_eq!(CriterionType::tier(), Tier::T1Universal);
    }

    #[test]
    fn direction_is_comparison_dominant() {
        let comp = Direction::primitive_composition();
        assert_eq!(comp.dominant, Some(LexPrimitiva::Comparison));
    }

    #[test]
    fn split_point_is_t2c() {
        assert_eq!(SplitPoint::tier(), Tier::T2Composite);
        assert_eq!(
            SplitPoint::primitive_composition().dominant,
            Some(LexPrimitiva::Comparison)
        );
    }

    #[test]
    fn tree_config_is_boundary_dominant() {
        let comp = TreeConfig::primitive_composition();
        assert_eq!(comp.dominant, Some(LexPrimitiva::Boundary));
    }

    #[test]
    fn prediction_result_is_t3() {
        assert_eq!(PredictionResult::tier(), Tier::T3DomainSpecific);
    }

    #[test]
    fn regression_result_is_quantity_dominant() {
        let comp = RegressionResult::primitive_composition();
        assert_eq!(comp.dominant, Some(LexPrimitiva::Quantity));
    }

    #[test]
    fn tree_node_is_recursion_dominant_t3() {
        assert_eq!(TreeNode::tier(), Tier::T3DomainSpecific);
        assert_eq!(
            TreeNode::primitive_composition().dominant,
            Some(LexPrimitiva::Recursion)
        );
    }

    #[test]
    fn feature_importance_is_t2p() {
        assert_eq!(FeatureImportance::tier(), Tier::T2Primitive);
    }

    #[test]
    fn tree_stats_is_t2c() {
        assert_eq!(TreeStats::tier(), Tier::T2Composite);
    }
}
