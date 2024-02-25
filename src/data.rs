use smartcore::{dataset::{breast_cancer, Dataset}, linalg::{naive::dense_matrix::DenseMatrix, BaseMatrix}, math::num::RealNumber};
use std::{mem, process::Output};
use sys_info;
pub trait DataSet {
    type Output;
    type Input: RealNumber + std::ops::Sub<Output = Self::Input>;
    // Internal dataset type that should be generic, it should be able to support many different concrete data representations and be agnostic to user
    type DataSetType;

    // all dataset but underlying the hood is operated on smartcore::DenseMatrix for local dataset
    fn default() -> Self;
    fn features(&self) -> Vec<String>;
    fn dimension(&self) -> (usize, usize);
    // return size in bytes
    fn size(&self) -> usize;
    fn fit_in_memory(&self) -> bool;
    fn target(&self) -> Vec<Self::Input>;
    fn select_columns(&self, column_selector: &[bool]) -> Option<DenseMatrix<Self::Input>> where <Self as DataSet>::Input: RealNumber;

    fn split_for_cross_validation(&self, k_folds: usize, fold: usize) -> (Self, Self) where Self: Sized;

    // fn data(&self) -> Self::DataSetType;
}
pub struct BreastCancerData {
    data: Dataset<f32, f32>
}

impl DataSet for BreastCancerData {
    type Input = f32;
    type Output = f32;
    type DataSetType = smartcore::dataset::Dataset<Self::Input, Self::Output>;

    fn default() -> Self {
        BreastCancerData {
            data: breast_cancer::load_dataset()
        }
        
    }

    fn features(&self) -> Vec<String> {
        self.data.feature_names.clone()
    }

    // (m, n)
    fn dimension(&self) -> (usize, usize) {
        (self.data.num_samples.clone(), self.data.num_features.clone())
    }

    // estimate bytes of dataset
    fn size(&self) -> usize {
        let data_size = self.data.data.len() * mem::size_of::<Self::Input>();
        let target_size = self.data.target.len() * mem::size_of::<Self::Output>();

        data_size + target_size
    }

    fn fit_in_memory(&self) -> bool {
        let ds_size_in_kb = self.size() / 1024;
        let mem_info = sys_info::mem_info().unwrap();
        let avail_mem = mem_info.avail + mem_info.free;
        if avail_mem > ds_size_in_kb as u64 {
            return true;
        }
        false
    }

    fn select_columns(&self, column_selector: &[bool]) -> Option<DenseMatrix<Self::Input>> {
        let selected_features: Vec<usize> = column_selector
            .iter()
            .enumerate()
            .filter_map(|(index, &feature)| if feature { Some(index) } else { None })
            .collect();
    
        if selected_features.is_empty() {
            return None;
        }
    
        let (m, _) = self.dimension(); // Original number of rows
    
        // Calculate the correct number of columns after selection
        let ncols = selected_features.len();
    
        let mut x_selected_data: Vec<f32> = Vec::with_capacity(m * ncols);
    
        for row in 0..m {
            for &col in &selected_features {
                let index = row * self.dimension().1 + col; // Calculate the flat index
                if index < self.data.data.len() {
                    x_selected_data.push(self.data.data[index]);
                } else {
                    // Handle or log the error if the index is out of bounds
                    // For simplicity, you might add a dummy value, log an error, or skip
                    x_selected_data.push(0.0); // Adding a dummy value as a placeholder
                }
            }
        }
    
        Some(DenseMatrix::from_vec(m, ncols, &x_selected_data))
    }
    

    fn target(&self) -> Vec<Self::Input> {
        self.data.target.clone()
    }

    fn split_for_cross_validation(&self, k_folds: usize, fold: usize) -> (Self, Self) {
        let (num_samples, num_features) = self.dimension();
        let fold_size = num_samples / k_folds;
        let remainder = num_samples % k_folds;
        let start_idx;
        let mut end_idx;
    
        // Adjust start and end index for each fold to distribute remainder samples
        if fold < remainder {
            // Folds that receive an extra sample
            start_idx = fold * (fold_size + 1);
            end_idx = start_idx + fold_size + 1;
        } else {
            // Folds with the regular number of samples
            start_idx = fold * fold_size + remainder;
            end_idx = start_idx + fold_size;
        }
    
        // Ensuring the end index does not exceed the total number of samples
        end_idx = end_idx.min(num_samples);
    
        // Create training dataset by excluding the range dedicated to the validation set
        let train_data = BreastCancerData {
            data: Dataset {
                data: self.data.data.iter().enumerate().filter_map(|(i, x)| if i < start_idx || i >= end_idx { Some(*x) } else { None }).collect(),
                target: self.data.target.iter().enumerate().filter_map(|(i, x)| if i < start_idx || i >= end_idx { Some(*x) } else { None }).collect(),
                feature_names: self.data.feature_names.clone(),
                target_names: self.data.target_names.clone(),
                description: self.data.description.clone(),
                num_samples: num_samples - (end_idx - start_idx), // Update based on excluded validation range
                num_features,
            }
        };
    
        // Create validation dataset from the specified range
        let valid_data = BreastCancerData {
            data: Dataset {
                data: self.data.data.iter().enumerate().filter_map(|(i, x)| if i >= start_idx && i < end_idx { Some(*x) } else { None }).collect(),
                target: self.data.target.iter().enumerate().filter_map(|(i, x)| if i >= start_idx && i < end_idx { Some(*x) } else { None }).collect(),
                feature_names: self.data.feature_names.clone(),
                target_names: self.data.target_names.clone(),
                description: self.data.description.clone(),
                num_samples: end_idx - start_idx, // Direct calculation for the validation range
                num_features,
            }
        };
    
        (train_data, valid_data)
    }
    

    // debug
    // fn data(&self) -> Self::DataSetType {
    //    self.data
    // }
    
}