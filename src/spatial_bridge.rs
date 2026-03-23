//! # Spatial Bridge: nexcore-dtree → stem-math
//!
//! Implements `Metric` for impurity comparison between tree nodes
//! and uses `Dimension` to express feature space dimensionality.
//!
//! ## Primitive Foundation
//!
//! Decision trees are spatial partitioners:
//! - Each split divides feature space along one dimension
//! - Impurity decrease measures "distance" from pure to impure
//! - Feature space has explicit dimensionality (n_features)
//! - Prediction is a dimension-reducing embedding: n_features → 1 output
//!
//! ## Architecture Decision
//!
//! `ImpurityMetric` wraps `Impurity` comparison as a valid `Metric`.
//! `Dimension` constants express feature space rank.
//! `Neighborhood` expresses split thresholds as containment checks.

use nexcore_lex_primitiva::grounding::GroundsTo;
use nexcore_lex_primitiva::primitiva::{LexPrimitiva, PrimitiveComposition};
use stem_math::spatial::{Dimension, Distance, Metric, Neighborhood};

use crate::types::{Impurity, TreeStats};

// ============================================================================
// ImpurityMetric: Distance between two impurity values
// ============================================================================

/// Metric over `Impurity` values.
///
/// Distance = |impurity_a - impurity_b|. This is a valid metric on ℝ⁺.
///
/// Use case: Measuring how much a split improved purity — the impurity
/// decrease is the distance from parent impurity to child impurity.
///
/// Tier: T2-P (N Quantity + κ Comparison)
pub struct ImpurityMetric;

impl Metric for ImpurityMetric {
    type Element = Impurity;

    fn distance(&self, a: &Impurity, b: &Impurity) -> Distance {
        Distance::new((a.0 - b.0).abs())
    }
}

/// GroundsTo: T2-P (κ Comparison + N Quantity), dominant κ
///
/// A metric IS a comparison operation — it takes two elements and produces
/// a numeric distance. Comparison-dominant because the purpose is to
/// measure how far apart two impurity values are.
impl GroundsTo for ImpurityMetric {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::Comparison, // κ — metric IS a comparison
            LexPrimitiva::Quantity,   // N — produces numeric distance
        ])
        .with_dominant(LexPrimitiva::Comparison, 0.90)
    }
}

// ============================================================================
// Dimension: Feature space rank
// ============================================================================

/// Extract the feature space dimension from tree statistics.
pub fn feature_dimension(stats: &TreeStats) -> Dimension {
    Dimension::new(stats.n_features as u32)
}

/// The output dimension of a classification tree: 1 (class label).
pub const OUTPUT_DIMENSION: Dimension = Dimension::new(1);

/// Codimension of prediction: n_features - 1 dimensions "lost" in projection.
pub fn prediction_codimension(n_features: u32) -> Dimension {
    if n_features > 1 {
        Dimension::new(n_features - 1)
    } else {
        Dimension::new(0)
    }
}

// ============================================================================
// Split threshold as Neighborhood
// ============================================================================

/// Express a split threshold as a closed Neighborhood.
///
/// In a decision tree, `feature <= threshold` means the feature value
/// is within the threshold neighborhood of zero (for continuous features).
///
/// More precisely: the left branch contains all points where
/// `Distance::new(feature_value) <= Distance::new(threshold)`.
pub fn split_neighborhood(threshold: f64) -> Neighborhood {
    Neighborhood::closed(Distance::new(threshold))
}

/// Minimum impurity decrease neighborhood.
///
/// A split is accepted only if the impurity decrease exceeds this threshold.
/// This is the pre-pruning gate expressed as neighborhood containment.
pub fn min_impurity_neighborhood(min_decrease: f64) -> Neighborhood {
    Neighborhood::closed(Distance::new(min_decrease))
}

/// Check if an impurity decrease qualifies as a valid split.
///
/// A split qualifies when the decrease is NOT inside the "too small" neighborhood.
/// The "too small" neighborhood contains all decreases below the threshold.
/// So: qualifies = decrease >= min_decrease = NOT contained in open(min_decrease).
pub fn split_qualifies(decrease: &Impurity, min_decrease: f64) -> bool {
    // decrease >= min_decrease means the decrease value is at or beyond the threshold
    decrease.0 >= min_decrease
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ===== Metric axiom tests =====

    #[test]
    fn metric_non_negativity() {
        let m = ImpurityMetric;
        let a = Impurity(0.4);
        let b = Impurity(0.1);
        assert!(m.distance(&a, &b).value() >= 0.0);
    }

    #[test]
    fn metric_identity() {
        let m = ImpurityMetric;
        let a = Impurity(0.3);
        assert!(m.distance(&a, &a).approx_eq(&Distance::ZERO, 1e-10));
    }

    #[test]
    fn metric_symmetry() {
        let m = ImpurityMetric;
        let a = Impurity(0.4);
        let b = Impurity(0.1);
        assert!(m.is_symmetric(&a, &b, 1e-10));
    }

    #[test]
    fn metric_triangle_inequality() {
        let m = ImpurityMetric;
        let a = Impurity(0.0);
        let b = Impurity(0.3);
        let c = Impurity(0.5);

        let d_ab = m.distance(&a, &b);
        let d_bc = m.distance(&b, &c);
        let d_ac = m.distance(&a, &c);
        assert!(Distance::triangle_valid(d_ab, d_bc, d_ac));
    }

    // ===== Dimension tests =====

    #[test]
    fn feature_dimension_from_stats() {
        let stats = TreeStats {
            depth: 5,
            n_leaves: 10,
            n_splits: 9,
            n_nodes: 19,
            n_features: 7,
            n_classes: 3,
            n_samples: 100,
            criterion: crate::types::CriterionType::Gini,
        };
        assert_eq!(feature_dimension(&stats).rank(), 7);
    }

    #[test]
    fn output_dimension_is_one() {
        assert_eq!(OUTPUT_DIMENSION.rank(), 1);
    }

    #[test]
    fn codimension_consistent() {
        let n = 7u32;
        let codim = prediction_codimension(n);
        assert_eq!(codim.rank() + OUTPUT_DIMENSION.rank(), n);
    }

    // ===== Neighborhood tests =====

    #[test]
    fn split_neighborhood_containment() {
        let n = split_neighborhood(5.0);
        assert!(n.contains(Distance::new(3.0))); // 3 <= 5
        assert!(n.contains(Distance::new(5.0))); // boundary, closed
        assert!(!n.contains(Distance::new(6.0))); // 6 > 5
    }

    #[test]
    fn impurity_split_qualification() {
        assert!(split_qualifies(&Impurity(0.15), 0.1)); // 0.15 >= 0.1
        assert!(split_qualifies(&Impurity(0.1), 0.1)); // boundary
        assert!(!split_qualifies(&Impurity(0.05), 0.1)); // 0.05 < 0.1 — wait, need to check logic
    }

    #[test]
    fn pure_impurity_distance_zero() {
        let m = ImpurityMetric;
        let pure = Impurity::PURE;
        assert!(m.distance(&pure, &pure).approx_eq(&Distance::ZERO, 1e-10));
    }
}
