use std::{ sync::Arc};
use fastgen::{
    data::{BreastCancerData, DataSet}, ga::{evaluate_fitness, run_ga, Individual}, model::{LinearRegressionModel, Model, ModelName}
};

// Ensure run_ga is imported or accessible here
// use fastgen::run_ga;

fn main() {
    // Load data
    let data = BreastCancerData::default();
    let num_features = data.dimension().1;

    // Define model
    // let model:  LinearRegressionModel = LinearRegressionModel::default();

    // Wrap data and model in Arc for shared access across threads
    let data_arc = Arc::new(data);
    // benchmark
    let individual = Individual::new_all_true(num_features);
    let res = evaluate_fitness::<BreastCancerData>(&individual, &data_arc, ModelName::LinearRegression);
    println!("without selection: {:?}", res);
    // Feature selection using the genetic algorithm
    // Assuming run_ga expects Arc-wrapped dataset and model, and returns results or updates state in-place
    run_ga::<BreastCancerData>(data_arc, ModelName::LinearRegression);

    // Depending on your implementation of run_ga, you may wish to retrieve and display results here.
    // For example, if run_ga mutates the model or has side effects you wish to inspect...
}

