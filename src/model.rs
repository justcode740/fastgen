use smartcore::{
    linalg::naive::dense_matrix::DenseMatrix,
    linear::linear_regression::{LinearRegression, LinearRegressionParameters},
    math::num::RealNumber,
    metrics::mean_squared_error,
};

#[derive(Clone)]
pub enum ModelName {
    LinearRegression,
}

use crate::data::DataSet;

pub trait Model<D: DataSet> {
    // Adjust methods to use D::Input and D::Output
    // TODO
    fn fit(
        x: DenseMatrix<D::Input>,
        y: Vec<D::Input>,
    ) -> Result<LinearRegression<f32, DenseMatrix<f32>>, String>;
    // fn predict(x: DenseMatrix<D::Input>) -> Result<Vec<D::Output>, String>;
    // fn evaluate(actual: &[D::Input], predicted: &[D::Output]) -> Result<f32, String>;
}

pub struct LinearRegressionModel
// where
//     D: DataSet<Input = f64, Output = f64>, // Constraint for simplicity
{
    model: Option<LinearRegression<f32, DenseMatrix<f32>>>, // Storing the model
}

impl Clone for LinearRegressionModel {
    fn clone(&self) -> Self {
        LinearRegressionModel { model: None }
    }
}

impl Default for LinearRegressionModel {
    fn default() -> Self {
        Self {
            model: Default::default(),
        }
    }
}

impl<D> Model<D> for LinearRegressionModel
where
    D: DataSet<Input = f32, Output = f32>, // Ensure D::Input and D::Output are f64
{
    fn fit(
        x: DenseMatrix<D::Input>,
        y: Vec<D::Output>,
    ) -> Result<LinearRegression<f32, DenseMatrix<f32>>, String> {
        let lr = LinearRegression::fit(&x, &y, Default::default()).map_err(|e| e.to_string())?;

        Ok(lr)
    }

    // fn predict(x: DenseMatrix<D::Input>) -> Result<Vec<D::Output>, String> {
    //     model.predict(&x).map_err(|e| e.to_string());

    // }

    // fn evaluate(&self, actual: &[D::Output], predicted: &[D::Output]) -> Result<f32, String> {
    //     if actual.len() != predicted.len() {
    //         return Err("Actual and predicted lengths do not match!".to_string());
    //     }
    //     let mse = mean_squared_error(&actual.to_vec(), &predicted.to_vec());
    //     Ok(mse)
    // }
}
