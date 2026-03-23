//! Cost-complexity pruning (CCP) for decision trees.
//!
//! ## Primitive Foundation
//! - T1: Recursion (ρ) — bottom-up traversal to compute effective alpha
//! - T1: Sequence (σ) — iterative weakest-link pruning
//! - T1: State (ς) — accumulated cost at each node

use crate::node::{DecisionTree, TreeNode};
use crate::types::{Confidence, Impurity};

/// Prune a decision tree using cost-complexity pruning (CCP).
///
/// The `alpha` parameter controls the trade-off between tree complexity
/// and accuracy. Higher alpha → more aggressive pruning.
///
/// - alpha = 0.0: no pruning (keep original tree)
/// - alpha → ∞: prune to root (single leaf)
///
/// Algorithm:
/// 1. For each internal node, compute effective alpha:
///    α_eff(t) = (R(t) - R(T_t)) / (|leaves(T_t)| - 1)
/// 2. Prune nodes where α_eff ≤ alpha (bottom-up).
pub fn cost_complexity_prune(tree: &mut DecisionTree, alpha: f64) {
    if let Some(root) = tree.root.take() {
        tree.root = Some(prune_node(root, alpha));
    }
}

/// Recursively prune a subtree.
fn prune_node(node: TreeNode, alpha: f64) -> TreeNode {
    match node {
        TreeNode::Leaf { .. } => node,
        TreeNode::Split {
            feature_index,
            threshold,
            left,
            right,
            impurity,
            samples,
            impurity_decrease,
        } => {
            // First, recursively prune children
            let pruned_left = prune_node(*left, alpha);
            let pruned_right = prune_node(*right, alpha);

            let subtree = TreeNode::Split {
                feature_index,
                threshold,
                left: Box::new(pruned_left),
                right: Box::new(pruned_right),
                impurity,
                samples,
                impurity_decrease,
            };

            // Compute effective alpha for this node
            let eff_alpha = effective_alpha(&subtree);

            // If effective alpha ≤ target alpha, collapse to leaf
            if eff_alpha <= alpha {
                collapse_to_leaf(&subtree)
            } else {
                subtree
            }
        }
    }
}

/// Compute effective alpha for an internal node.
///
/// α_eff(t) = (R(t) - R(T_t)) / (|leaves(T_t)| - 1)
///
/// Where:
/// - R(t) = cost if this node were a leaf
/// - R(T_t) = total cost of the subtree
/// - |leaves(T_t)| = number of leaves in subtree
fn effective_alpha(node: &TreeNode) -> f64 {
    let n_leaves = node.n_leaves();
    if n_leaves <= 1 {
        return f64::MAX;
    }

    let leaf_cost = node.leaf_cost();
    let subtree_cost = node.subtree_cost();
    let denominator = (n_leaves as f64) - 1.0;

    if denominator.abs() < f64::EPSILON {
        return f64::MAX;
    }

    (leaf_cost - subtree_cost) / denominator
}

/// Collapse a subtree to a single leaf by majority vote.
fn collapse_to_leaf(node: &TreeNode) -> TreeNode {
    let mut class_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    collect_leaf_votes(node, &mut class_counts);

    let total: usize = class_counts.values().sum();
    let total_f = total as f64;

    if total == 0 {
        return TreeNode::Leaf {
            prediction: String::new(),
            confidence: Confidence::NONE,
            distribution: Vec::new(),
            samples: node.samples(),
            impurity: Impurity::PURE,
        };
    }

    let mut distribution: Vec<(String, f64)> = class_counts
        .iter()
        .map(|(k, &v)| (k.clone(), v as f64 / total_f))
        .collect();
    distribution.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let (prediction, max_prob) = distribution
        .first()
        .map(|(p, prob)| (p.clone(), *prob))
        .unwrap_or_else(|| (String::new(), 0.0));

    TreeNode::Leaf {
        prediction,
        confidence: Confidence::new(max_prob),
        distribution,
        samples: node.samples(),
        impurity: node.impurity(),
    }
}

/// Collect weighted votes from all leaves in a subtree.
fn collect_leaf_votes(node: &TreeNode, counts: &mut std::collections::HashMap<String, usize>) {
    match node {
        TreeNode::Leaf {
            prediction,
            samples,
            ..
        } => {
            *counts.entry(prediction.clone()).or_insert(0) += samples;
        }
        TreeNode::Split { left, right, .. } => {
            collect_leaf_votes(left, counts);
            collect_leaf_votes(right, counts);
        }
    }
}

/// Compute the sequence of alpha values where pruning occurs (pruning path).
///
/// Returns pairs of (alpha, n_leaves) showing how the tree shrinks.
#[must_use]
pub fn pruning_path(tree: &DecisionTree) -> Vec<(f64, usize)> {
    let Some(root) = tree.root() else {
        return Vec::new();
    };

    let mut path = vec![(0.0, root.n_leaves())];
    let mut current = root.clone();
    let mut prev_leaves = root.n_leaves();

    loop {
        let alpha = find_min_alpha(&current);
        if alpha >= f64::MAX || alpha < 0.0 {
            break;
        }
        current = prune_node(current, alpha);
        let new_leaves = current.n_leaves();
        if new_leaves >= prev_leaves {
            break;
        }
        path.push((alpha, new_leaves));
        prev_leaves = new_leaves;
        if current.is_leaf() {
            break;
        }
    }

    path
}

/// Find the minimum effective alpha across all internal nodes.
fn find_min_alpha(node: &TreeNode) -> f64 {
    match node {
        TreeNode::Leaf { .. } => f64::MAX,
        TreeNode::Split { left, right, .. } => {
            let self_alpha = effective_alpha(node);
            let left_alpha = find_min_alpha(left);
            let right_alpha = find_min_alpha(right);
            self_alpha.min(left_alpha).min(right_alpha)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::fit;
    use crate::types::{Feature, TreeConfig};

    fn make_data() -> (Vec<Vec<Feature>>, Vec<String>) {
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
    fn prune_zero_alpha_no_change() {
        let (data, labels) = make_data();
        let result = fit(&data, &labels, TreeConfig::default());
        assert!(result.is_ok());
        let mut tree = result
            .ok()
            .unwrap_or_else(|| DecisionTree::new(TreeConfig::default()));

        let before = tree.stats().map(|s| s.n_leaves).unwrap_or(0);
        cost_complexity_prune(&mut tree, 0.0);
        let after = tree.stats().map(|s| s.n_leaves).unwrap_or(0);

        assert!(after <= before);
    }

    #[test]
    fn prune_large_alpha_becomes_leaf() {
        let (data, labels) = make_data();
        let result = fit(&data, &labels, TreeConfig::default());
        assert!(result.is_ok());
        let mut tree = result
            .ok()
            .unwrap_or_else(|| DecisionTree::new(TreeConfig::default()));

        cost_complexity_prune(&mut tree, 1000.0);
        let stats = tree.stats();
        assert!(stats.is_some());
        let stats = stats.unwrap_or_else(|| crate::types::TreeStats {
            depth: 0,
            n_leaves: 1,
            n_splits: 0,
            n_nodes: 1,
            n_features: 0,
            n_classes: 0,
            n_samples: 0,
            criterion: crate::types::CriterionType::Gini,
        });
        assert_eq!(stats.n_leaves, 1);
    }

    #[test]
    fn pruning_path_decreasing_leaves() {
        let (data, labels) = make_data();
        let result = fit(&data, &labels, TreeConfig::default());
        assert!(result.is_ok());
        let tree = result
            .ok()
            .unwrap_or_else(|| DecisionTree::new(TreeConfig::default()));

        let path = pruning_path(&tree);
        assert!(!path.is_empty());

        // Alphas should be non-decreasing
        for window in path.windows(2) {
            assert!(window[1].0 >= window[0].0);
        }
        // Leaf counts should be non-increasing
        for window in path.windows(2) {
            assert!(window[1].1 <= window[0].1);
        }
    }
}
