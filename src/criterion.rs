//! Pluggable splitting criteria for decision tree training.
//!
//! ## Primitive Foundation
//! - T1: Mapping (μ) — class counts → impurity score
//! - T1: Sequence (σ) — iterate over class counts
//!
//! ## Supported Criteria
//! - **Gini**: 1 - Σ(p_i²) — CART default for classification
//! - **Entropy**: -Σ(p_i * log2(p_i)) — ID3/C4.5
//! - **GainRatio**: IG / SplitInfo — C4.5 correction for high-cardinality features
//! - **MSE**: Σ(y_i - ȳ)² / n — Regression trees

use crate::types::{CriterionType, Impurity};

/// Trait for computing node impurity from class distributions.
///
/// Tier: T2-C (trait abstraction over T1 sequence + mapping)
pub trait SplitCriterion: Send + Sync {
    /// Compute impurity given class counts and total samples.
    ///
    /// `class_counts[i]` = number of samples in class i.
    /// `total` = sum of all class counts.
    fn impurity(&self, class_counts: &[usize], total: usize) -> Impurity;

    /// Human-readable name of this criterion.
    fn name(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// Gini Impurity
// ---------------------------------------------------------------------------

/// Gini impurity: 1 - Σ(p_i²)
///
/// Measures the probability of misclassification. Pure node = 0.
/// Maximum for C classes: 1 - 1/C.
///
/// Tier: T2-C
pub struct Gini;

impl SplitCriterion for Gini {
    fn impurity(&self, class_counts: &[usize], total: usize) -> Impurity {
        if total == 0 {
            return Impurity::PURE;
        }
        let total_f = total as f64;
        let sum_sq: f64 = class_counts
            .iter()
            .map(|&c| {
                let p = c as f64 / total_f;
                p * p
            })
            .sum();
        Impurity(1.0 - sum_sq)
    }

    fn name(&self) -> &'static str {
        "gini"
    }
}

// ---------------------------------------------------------------------------
// Shannon Entropy
// ---------------------------------------------------------------------------

/// Shannon entropy: -Σ(p_i * log2(p_i))
///
/// Information-theoretic impurity. Pure node = 0.
/// Maximum for C classes: log2(C).
///
/// Tier: T2-C
pub struct Entropy;

impl SplitCriterion for Entropy {
    fn impurity(&self, class_counts: &[usize], total: usize) -> Impurity {
        if total == 0 {
            return Impurity::PURE;
        }
        let total_f = total as f64;
        let entropy: f64 = class_counts
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / total_f;
                -p * p.log2()
            })
            .sum();
        Impurity(entropy)
    }

    fn name(&self) -> &'static str {
        "entropy"
    }
}

// ---------------------------------------------------------------------------
// Gain Ratio
// ---------------------------------------------------------------------------

/// Gain ratio: IG / SplitInfo
///
/// C4.5 correction that penalizes high-cardinality features.
/// For single-node impurity, this is identical to Entropy.
/// The ratio correction is applied in the split evaluation, not here.
///
/// Tier: T2-C
pub struct GainRatio;

impl SplitCriterion for GainRatio {
    fn impurity(&self, class_counts: &[usize], total: usize) -> Impurity {
        // Node impurity is same as entropy; ratio applied during split selection.
        Entropy.impurity(class_counts, total)
    }

    fn name(&self) -> &'static str {
        "gain_ratio"
    }
}

/// Compute split information for gain ratio correction.
///
/// SplitInfo = -Σ((n_j / N) * log2(n_j / N)) where j ∈ {left, right}.
#[must_use]
pub fn split_info(left_count: usize, right_count: usize) -> f64 {
    let total = (left_count + right_count) as f64;
    if total == 0.0 {
        return 0.0;
    }

    let mut info = 0.0;
    for &count in &[left_count, right_count] {
        if count > 0 {
            let p = count as f64 / total;
            info -= p * p.log2();
        }
    }
    info
}

// ---------------------------------------------------------------------------
// Mean Squared Error (Regression)
// ---------------------------------------------------------------------------

/// Mean squared error: Σ(y_i - ȳ)² / n
///
/// For regression trees. Lower is better.
///
/// Tier: T2-C
pub struct Mse;

impl Mse {
    /// Compute MSE from raw target values.
    #[must_use]
    pub fn impurity_from_values(values: &[f64]) -> Impurity {
        if values.is_empty() {
            return Impurity::PURE;
        }
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let mse = values.iter().map(|&y| (y - mean).powi(2)).sum::<f64>() / n;
        Impurity(mse)
    }

    /// Compute variance from sum and sum-of-squares (online formula).
    #[must_use]
    pub fn impurity_from_stats(sum: f64, sum_sq: f64, count: usize) -> Impurity {
        if count == 0 {
            return Impurity::PURE;
        }
        let n = count as f64;
        let mean = sum / n;
        let variance = sum_sq / n - mean * mean;
        // Clamp to handle floating-point drift
        Impurity(variance.max(0.0))
    }
}

impl SplitCriterion for Mse {
    fn impurity(&self, class_counts: &[usize], total: usize) -> Impurity {
        // MSE cannot be computed from class counts alone.
        // This trait method is a fallback; use `impurity_from_values` instead.
        // Return a simple diversity measure as proxy.
        if total == 0 {
            return Impurity::PURE;
        }
        Gini.impurity(class_counts, total)
    }

    fn name(&self) -> &'static str {
        "mse"
    }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/// Create a boxed `SplitCriterion` from a `CriterionType`.
#[must_use]
pub fn make_criterion(criterion_type: CriterionType) -> Box<dyn SplitCriterion> {
    match criterion_type {
        CriterionType::Gini => Box::new(Gini),
        CriterionType::Entropy => Box::new(Entropy),
        CriterionType::GainRatio => Box::new(GainRatio),
        CriterionType::Mse => Box::new(Mse),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gini_pure_node() {
        let imp = Gini.impurity(&[10, 0], 10);
        assert!(imp.is_pure());
    }

    #[test]
    fn gini_balanced_binary() {
        // Two classes, 50/50 → Gini = 0.5
        let imp = Gini.impurity(&[5, 5], 10);
        assert!((imp.0 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn gini_three_classes_uniform() {
        // Three classes, uniform → Gini = 1 - 3*(1/3)² = 1 - 1/3 = 2/3
        let imp = Gini.impurity(&[10, 10, 10], 30);
        assert!((imp.0 - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn entropy_pure_node() {
        let imp = Entropy.impurity(&[10, 0], 10);
        assert!(imp.is_pure());
    }

    #[test]
    fn entropy_balanced_binary() {
        // Two classes, 50/50 → Entropy = 1.0 bit
        let imp = Entropy.impurity(&[5, 5], 10);
        assert!((imp.0 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn entropy_three_classes_uniform() {
        // Three classes, uniform → Entropy = log2(3) ≈ 1.585
        let imp = Entropy.impurity(&[10, 10, 10], 30);
        assert!((imp.0 - 3.0_f64.log2()).abs() < 1e-10);
    }

    #[test]
    fn mse_from_values() {
        // Values: [1, 2, 3], mean=2, MSE = ((1-2)²+(2-2)²+(3-2)²)/3 = 2/3
        let imp = Mse::impurity_from_values(&[1.0, 2.0, 3.0]);
        assert!((imp.0 - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn mse_pure() {
        let imp = Mse::impurity_from_values(&[5.0, 5.0, 5.0]);
        assert!(imp.is_pure());
    }

    #[test]
    fn split_info_balanced() {
        // 50/50 split → SplitInfo = 1.0 bit
        let si = split_info(50, 50);
        assert!((si - 1.0).abs() < 1e-10);
    }

    #[test]
    fn split_info_skewed() {
        // 90/10 split → SplitInfo ≈ 0.469
        let si = split_info(90, 10);
        assert!(si > 0.4 && si < 0.5);
    }

    #[test]
    fn gini_empty() {
        let imp = Gini.impurity(&[], 0);
        assert!(imp.is_pure());
    }

    #[test]
    fn entropy_empty() {
        let imp = Entropy.impurity(&[], 0);
        assert!(imp.is_pure());
    }

    #[test]
    fn make_criterion_factory() {
        let c = make_criterion(CriterionType::Gini);
        assert_eq!(c.name(), "gini");

        let c = make_criterion(CriterionType::Entropy);
        assert_eq!(c.name(), "entropy");

        let c = make_criterion(CriterionType::GainRatio);
        assert_eq!(c.name(), "gain_ratio");

        let c = make_criterion(CriterionType::Mse);
        assert_eq!(c.name(), "mse");
    }
}
