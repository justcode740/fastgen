use std::{ sync::Arc};
use fastgen::{
    config::GaConfig, data::{BreastCancerData, DataSet}, ga::{evaluate_fitness, run_ga, run_ga_cross_validation, Individual}, model::{LinearRegressionModel, Model, ModelName}
};

// Ensure run_ga is imported or accessible here
// use fastgen::run_ga;
fn main() {
    // Load data
    let data = BreastCancerData::default();

    // Wrap data in Arc for shared access across threads
    let data_arc = Arc::new(data);

    // Define GA configuration
    let ga_config = GaConfig {
        generations: 20,
        populaton_size: 50,
    };

    // Run GA with cross-validation
    let k_folds = 2; // Number of folds for cross-validation
    let (avg_mse, best_features) = run_ga_cross_validation(data_arc.clone(), ModelName::LinearRegression, ga_config, k_folds);

    println!("Average MSE across {} folds: {}", k_folds, avg_mse);
    println!("Best feature selection: {:?}", best_features);

    // For benchmarking without feature selection, you can simulate a train-validation split by manually selecting a portion of your dataset
    // This example assumes your dataset can be manually indexed or split; you might need to adjust this based on your dataset's structure
    // For demonstration, this part is conceptual and might need adaptation
    let num_samples = data_arc.dimension().0;
    let num_features = data_arc.dimension().1;
    let individual = Individual::new_all_true(num_features);
    // Assuming a conceptual split, replace with actual logic as per your dataset's implementation
    let validation_fold = 0; // Use the first fold as a proxy for validation
    let (train_set, valid_set) = data_arc.split_for_cross_validation(k_folds, validation_fold);
    let validation_mse = evaluate_fitness::<BreastCancerData>(&individual, &Arc::new(valid_set), ModelName::LinearRegression);
    println!("Validation set MSE (Benchmark, no feature selection): {}", validation_mse);
}


