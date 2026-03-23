use nexcore_dtree::prelude::*;

#[test]
fn test_biomarker_validity_classification() {
    // 1. DATA PREPARATION: Biomarker Validation Features
    // Feature index mapping:
    // 0: Replication Success (1.0 = yes, 0.0 = no)
    // 1: Multi-cell type validation (1.0 = yes, 0.0 = no)
    // 2: Clinical trial validation (count)
    // 3: Effect Size (fold change)
    // 4: mRNA-Protein Concordance (1.0 = yes, 0.0 = no)

    let data = vec![
        // Valid Biomarkers (High confidence)
        vec![
            Feature::Continuous(1.0),
            Feature::Continuous(1.0),
            Feature::Continuous(3.0),
            Feature::Continuous(4.0),
            Feature::Continuous(1.0),
        ],
        vec![
            Feature::Continuous(1.0),
            Feature::Continuous(1.0),
            Feature::Continuous(2.0),
            Feature::Continuous(3.5),
            Feature::Continuous(1.0),
        ],
        // Promising (Intermediate)
        vec![
            Feature::Continuous(1.0),
            Feature::Continuous(1.0),
            Feature::Continuous(0.0),
            Feature::Continuous(2.0),
            Feature::Continuous(0.0),
        ],
        vec![
            Feature::Continuous(0.0),
            Feature::Continuous(1.0),
            Feature::Continuous(0.0),
            Feature::Continuous(5.0),
            Feature::Continuous(0.0),
        ],
        // Artifacts (Low confidence)
        vec![
            Feature::Continuous(0.0),
            Feature::Continuous(0.0),
            Feature::Continuous(0.0),
            Feature::Continuous(1.1),
            Feature::Continuous(0.0),
        ],
        vec![
            Feature::Continuous(0.0),
            Feature::Continuous(0.0),
            Feature::Continuous(0.0),
            Feature::Continuous(1.5),
            Feature::Continuous(0.0),
        ],
    ];

    let labels = vec![
        "Valid".to_string(),
        "Valid".to_string(),
        "Promising".to_string(),
        "Promising".to_string(),
        "Artifact".to_string(),
        "Artifact".to_string(),
    ];

    // 2. TRAINING
    let config = TreeConfig {
        max_depth: Some(5),
        min_samples_leaf: 1,
        ..TreeConfig::default()
    };
    let tree = fit(&data, &labels, config).expect("Failed to fit decision tree");

    println!("--- BIOMARKER VALIDITY MODEL ---");
    println!(
        "Tree trained on {} samples.",
        tree.stats().unwrap().n_samples
    );

    // 3. INFERENCE: Evaluating HEXIM1 as a Biomarker
    // Metrics from the HEXIM1 report:
    // - Replication: FAILED for SLE baseline (0.0)
    // - Multi-cell type: YES (Macrophages, NK, T cells, ILC2) (1.0)
    // - Clinical trials: 3 (1.0)
    // - Effect size: 4.13x (4.13)
    // - Protein concordance: YES (1.0)

    let hexim1_pd_features = vec![
        Feature::Continuous(1.0),  // High replication success overall for PD marker
        Feature::Continuous(1.0),  // Multi-cell
        Feature::Continuous(3.0),  // 3 trials
        Feature::Continuous(4.13), // 4.13x
        Feature::Continuous(1.0),  // Concordant
    ];

    let hexim1_sle_baseline_features = vec![
        Feature::Continuous(0.0),  // FAILED replication
        Feature::Continuous(0.0),  // Only one study
        Feature::Continuous(0.0),  // 0 trials
        Feature::Continuous(1.18), // 1.18x (small)
        Feature::Continuous(0.0),  // Unknown/None
    ];

    let pd_prediction = predict(&tree, &hexim1_pd_features).unwrap();
    let sle_prediction = predict(&tree, &hexim1_sle_baseline_features).unwrap();

    println!("\nHEXIM1 PD Marker Evaluation:");
    println!("  Prediction: {}", pd_prediction.prediction);
    println!("  Confidence: {:.2}", pd_prediction.confidence.value());

    println!("\nHEXIM1 SLE Baseline Evaluation:");
    println!("  Prediction: {}", sle_prediction.prediction);
    println!("  Confidence: {:.2}", sle_prediction.confidence.value());

    // 4. RULE EXTRACTION (The "Scientific Protocol")
    println!("\nGenerated Research Protocol (Tree Rules):");
    let rules = to_rules(&tree).expect("Failed to generate rules");
    println!("{}", rules);

    // 5. ASSERTIONS
    assert_eq!(pd_prediction.prediction, "Valid");
    assert_eq!(sle_prediction.prediction, "Artifact");
}
