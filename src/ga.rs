extern crate rayon;
extern crate rand;
extern crate smartcore;

use core::num;
use std::{any::Any, iter::Sum, sync::Arc};

use rand::Rng;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use smartcore::{linear::linear_regression::LinearRegression, metrics::mean_squared_error};


use crate::{data::{self, DataSet}, model::{LinearRegressionModel, Model, ModelName}};

#[derive(Clone, Debug)]
pub struct Individual {
    features: Vec<bool>,
    fitness: f32,
}

impl Individual {
    // new random individual
    fn new(num_features: usize) -> Self {
        let mut rng = rand::thread_rng();
        Self {
            features: (0..num_features).map(|_| rng.gen()).collect(),
            fitness: 0.0,
        }
    }

    // select all features
    pub fn new_all_true(num_features: usize) -> Self {
        Self {
            features: vec![true; num_features],
            fitness: 0.0,
        }
    }

    fn mutate(&mut self) {
        let mut rng = rand::thread_rng();
        for gene in self.features.iter_mut() {
            if rng.gen_bool(0.01) {
                *gene = !*gene;
            }
        }
    }
}

pub fn evaluate_fitness<D>(individual: &Individual, dataset: &D, model: ModelName) -> f32
where
    D: DataSet, // DataSet trait is assumed to provide select_columns and target methods.
    D::Input: std::ops::Sub<Output = D::Input>,
    f32: Sum<<D as DataSet>::Input> // Model trait as defined previously.
{
    match model {
        ModelName::LinearRegression => {
             // Select features based on the individual's characteristics
             if let Some(x_selected) = dataset.select_columns(&individual.features) {
                // Prepare the target data from the dataset
                let actual = dataset.target(); // Assuming this returns Vec<f64>, matching the DataSet trait

                // Create an instance of Linear Regression from SmartCore
                let lr = LinearRegression::fit(&x_selected, &actual, Default::default()).unwrap();

                // Predict outcomes using the same selected features
                if let Ok(predictions) = lr.predict(&x_selected) {
                    // Evaluate the model's predictions against the actual outcomes
                    let mse: f32 = actual
                    .iter()
                    .zip(predictions.iter())
                    .map(|(a, p)| (*a - *p) * (*a - *p)) // Subtracting actual from predicted values
                    .sum::<f32>()
                    / actual.len() as f32;
                    return -mse;
                    
                }

                // // Attempt to fit the model with the selected features and the actual outcomes
                // if lr.fit().is_ok() {
                    
                // }
            }
            0.0 // Default return value in case of any failure
            
        },
        _ => panic!(""),
    }


   
}


pub fn run_ga<D>(dataset: Arc<D>, model: ModelName)
where
    D: DataSet + Sync + Send, // Ensure the dataset is Sync to be shared across threads.
    f32: Sum<<D as DataSet>::Input>// Model needs to be Cloneable and Send to be used in parallel.
{
    let num_features = dataset.dimension().1;
    let population_size = 50;
    let generations = 20;

    let mut population: Vec<Individual> = (0..population_size)
        .map(|_| Individual::new(num_features))
        .collect();

    for _ in 0..generations {
        population.par_iter_mut().for_each(|individual| {
            // Initialize a new model for each thread/individual.
            // let mut model_instance = model.clone();
        
            // Dereference the Arc to get a shared reference to the dataset (D).
            let dataset_ref = Arc::as_ref(&dataset);
        
            // Pass the new model instance and the dataset reference to `evaluate_fitness`.
            individual.fitness = evaluate_fitness::<D>(individual, dataset_ref, model.clone());
        });

        // Sort the population by fitness, keep the best half, and mutate to form a new generation.
        population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal));
        let survivors = population.split_off(population_size / 2);
        let mut new_generation = survivors.clone();
        new_generation.iter_mut().for_each(|individual| {
            individual.mutate();
        });
        population = [population, new_generation].concat();
    }

    // Output the best individual's fitness and selected features.
    let best_individual = &population[0];
    println!("Best Fitness (Negative MSE): {}", best_individual.fitness);
    println!("Selected Features: {:?}", best_individual.features);
}