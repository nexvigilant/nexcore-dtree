//! Serialization: JSON export and human-readable rule format.
//!
//! ## Primitive Foundation
//! - T1: Recursion (ρ) — recursive tree traversal for rule generation
//! - T1: Mapping (μ) — node → JSON / rule string

use crate::node::{DecisionTree, TreeNode};
use crate::types::Feature;

/// Error type for serialization failures.
#[derive(Debug, nexcore_error::Error)]
pub enum SerializeError {
    /// Tree has not been fitted.
    #[error("tree has not been fitted")]
    NotFitted,
    /// JSON serialization error.
    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Export the tree to JSON format.
///
/// # Errors
/// Returns `SerializeError` if tree is not fitted or serialization fails.
pub fn to_json(tree: &DecisionTree) -> Result<String, SerializeError> {
    if !tree.is_fitted() {
        return Err(SerializeError::NotFitted);
    }
    let json = serde_json::to_string_pretty(tree)?;
    Ok(json)
}

/// Export the tree to compact JSON format.
///
/// # Errors
/// Returns `SerializeError` if tree is not fitted or serialization fails.
pub fn to_json_compact(tree: &DecisionTree) -> Result<String, SerializeError> {
    if !tree.is_fitted() {
        return Err(SerializeError::NotFitted);
    }
    let json = serde_json::to_string(tree)?;
    Ok(json)
}

/// Import a tree from JSON.
///
/// # Errors
/// Returns `SerializeError` on parse failure.
pub fn from_json(json: &str) -> Result<DecisionTree, SerializeError> {
    let tree: DecisionTree = serde_json::from_str(json)?;
    Ok(tree)
}

/// Export the tree as human-readable if-then-else rules.
///
/// # Errors
/// Returns `SerializeError` if tree is not fitted.
pub fn to_rules(tree: &DecisionTree) -> Result<String, SerializeError> {
    if !tree.is_fitted() {
        return Err(SerializeError::NotFitted);
    }

    let Some(root) = tree.root() else {
        return Err(SerializeError::NotFitted);
    };

    let mut output = String::new();
    format_rules(root, &tree.feature_names, 0, &mut output);
    Ok(output)
}

/// Recursively format tree nodes as if-then-else rules.
fn format_rules(node: &TreeNode, feature_names: &[String], indent: usize, output: &mut String) {
    let prefix = "  ".repeat(indent);
    match node {
        TreeNode::Leaf {
            prediction,
            confidence,
            samples,
            ..
        } => {
            output.push_str(&format!(
                "{prefix}PREDICT \"{prediction}\" (confidence={}, samples={samples})\n",
                confidence.value()
            ));
        }
        TreeNode::Split {
            feature_index,
            threshold,
            left,
            right,
            samples,
            ..
        } => {
            let feat_name = feature_names
                .get(*feature_index)
                .cloned()
                .unwrap_or_else(|| format!("feature[{feature_index}]"));
            let thresh_str = format_threshold(threshold);

            output.push_str(&format!(
                "{prefix}IF {feat_name} <= {thresh_str} (samples={samples}):\n"
            ));
            format_rules(left, feature_names, indent + 1, output);

            output.push_str(&format!("{prefix}ELSE ({feat_name} > {thresh_str}):\n"));
            format_rules(right, feature_names, indent + 1, output);
        }
    }
}

/// Format a threshold value for display.
fn format_threshold(feature: &Feature) -> String {
    match feature {
        Feature::Continuous(v) => format!("{v:.4}"),
        Feature::Categorical(s) => format!("\"{s}\""),
        Feature::Missing => "?".to_string(),
    }
}

/// Export tree statistics as a summary string.
///
/// # Errors
/// Returns `SerializeError` if tree is not fitted.
pub fn to_summary(tree: &DecisionTree) -> Result<String, SerializeError> {
    if !tree.is_fitted() {
        return Err(SerializeError::NotFitted);
    }

    let stats = tree.stats().ok_or(SerializeError::NotFitted)?;
    let summary = format!(
        "Decision Tree Summary\n\
         =====================\n\
         Criterion:  {:?}\n\
         Depth:      {}\n\
         Leaves:     {}\n\
         Splits:     {}\n\
         Nodes:      {}\n\
         Features:   {}\n\
         Classes:    {}\n\
         Samples:    {}\n",
        stats.criterion,
        stats.depth,
        stats.n_leaves,
        stats.n_splits,
        stats.n_nodes,
        stats.n_features,
        stats.n_classes,
        stats.n_samples,
    );
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::fit;
    use crate::types::{Feature, TreeConfig};

    fn make_tree() -> DecisionTree {
        let data = vec![
            vec![Feature::Continuous(0.1)],
            vec![Feature::Continuous(0.2)],
            vec![Feature::Continuous(0.8)],
            vec![Feature::Continuous(0.9)],
        ];
        let labels = vec!["A".into(), "A".into(), "B".into(), "B".into()];
        fit(&data, &labels, TreeConfig::default())
            .ok()
            .unwrap_or_else(|| DecisionTree::new(TreeConfig::default()))
    }

    #[test]
    fn json_roundtrip() {
        let tree = make_tree();
        let json = to_json(&tree);
        assert!(json.is_ok());
        let json_str = json.ok().unwrap_or_default();

        let restored = from_json(&json_str);
        assert!(restored.is_ok());
        let restored = restored
            .ok()
            .unwrap_or_else(|| DecisionTree::new(TreeConfig::default()));
        assert!(restored.is_fitted());
    }

    #[test]
    fn compact_json_is_shorter() {
        let tree = make_tree();
        let pretty = to_json(&tree).ok().unwrap_or_default();
        let compact = to_json_compact(&tree).ok().unwrap_or_default();
        assert!(compact.len() <= pretty.len());
    }

    #[test]
    fn rules_contain_if_else() {
        let tree = make_tree();
        let rules = to_rules(&tree).ok().unwrap_or_default();
        assert!(rules.contains("IF") || rules.contains("PREDICT"));
    }

    #[test]
    fn summary_contains_stats() {
        let tree = make_tree();
        let summary = to_summary(&tree).ok().unwrap_or_default();
        assert!(summary.contains("Depth:"));
        assert!(summary.contains("Leaves:"));
    }

    #[test]
    fn unfitted_tree_errors() {
        let tree = DecisionTree::new(TreeConfig::default());
        assert!(to_json(&tree).is_err());
        assert!(to_rules(&tree).is_err());
        assert!(to_summary(&tree).is_err());
    }
}
