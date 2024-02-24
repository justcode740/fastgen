use smartcore::{dataset::{breast_cancer, Dataset}, linalg::{naive::dense_matrix::DenseMatrix, BaseMatrix}, math::num::RealNumber};
use std::{mem, process::Output};
use sys_info;
pub trait DataSet {
    type Output;
    type Input: RealNumber + std::ops::Sub<Output = Self::Input>;
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
        // Filter out the indices of features that are selected (1)
        let selected_features: Vec<usize> = column_selector
        .iter()
        .enumerate()
        .filter_map(|(index, &feature)| if feature { Some(index) } else { None })
        .collect();

        // Early return if no features are selected
        if selected_features.is_empty() {
            return None;
        }
        let (m, n) = self.dimension();

        let x = DenseMatrix::from_array(m, n, &self.data.data);

        // Collect selected columns' values into a Vec<f32> and then create a new DenseMatrix
        let ncols = selected_features.len(); // Number of selected columns

        let mut x_selected_data: Vec<f32> = Vec::with_capacity(m * ncols);

        for row in 0..m {
            for &col in &selected_features {
                x_selected_data.push(x.get(row, col));
            }
        }

        // Create a new DenseMatrix with the selected features
        Some(
            DenseMatrix::from_vec(m, ncols, &x_selected_data)
        )
    }

    fn target(&self) -> Vec<Self::Input> {
        self.data.target.clone()
    }
}