# Blazing Fast, Generic Library for Automated Feature and Model Selection

This library is designed to offer a blazing-fast, generic solution for automated feature and model selection, tailored for large datasets. By leveraging the power of Rust's parallelism and abstraction, alongside advanced model selection techniques, it streamlines the process of preparing data and choosing the optimal machine learning model with the best feature sets for your needs.

## Experiment run
| Model                | GA Time (s)    | GA MSE      | Baseline MSE | Features (Baseline/GA) | Generations | Population Size | k-Folds |
|----------------------|----------------|-------------|--------------|------------------------|-------------|-----------------|---------|
| LinearRegression     | 16.51424405    | -0.20968826 | -0.24114878  | 30/16                  | 100         | 50              | 2       |
| DecisionTreeRegressor| 150.815264143  | -0.2097171  | -0.2413631   | 30/12                  | 100         | 50              | 2       |


## Key Features

### Data Parallelism and Dataset Abstraction
- Provides a **unified interface** for interacting with various datasets, whether part of `smartcore::dataset`, stored locally, or distributed across a network. This abstraction simplifies data handling and ensures efficient processing across different storage solutions.

### Model Parallelism
- Employs model parallelism to distribute the training of complex models across multiple computing nodes, significantly reducing training times and facilitating the handling of larger, more intricate models.

### Cross-Validation and Genetic Algorithms
- Utilizes cross-validation techniques to ensure the reliability and generalizability of the model evaluation process.
- Incorporates genetic algorithms for feature selection, offering a sophisticated approach to identifying the most effective features for model training.

### Parallelism and Pipelining Across Feature Selection Stages
- Enhances the efficiency of the feature selection process through parallelism and pipelining, ensuring the rapid identification of the most relevant features for your models.
![alt text](image.png)
## Roadmap (Todo)
- **Data Loading**: Enable actual loading of datasets from both network sources and local storage, with storage links for seamless data integration.
- **Model Generality and Flexibility**: Refactor the current system to make model handling truly generic, adding support for more machine learning models from various libraries.
- **Model Classification and Metrics**: Analyze and classify different models based on their performance, focusing on metrics-wise evaluation for better decision-making.
- **Advanced Genetic Algorithm Features**: Integrate sophisticated genetic algorithm features, such as elitism, to enhance the effectiveness and efficiency of the selection process.
- **Contributor-Friendly Modularity**: Continue to develop a modular codebase that encourages and simplifies contributions across various aspects of the library.
- **User Interface Enhancements**: Develop a more user-friendly CLI and output system, making the library more accessible and easier to use for a broader audience.
- **Frontend Visualization**: Explore the creation of a frontend interface to visualize the feature and model selection process, offering users insight into the decision-making behind the scenes.

## Modularity and Contributor Friendliness

Our library is built with **modularity** and **contributor friendliness** at its core. We believe that a modular design not only facilitates easier understanding and use of the software but also encourages community contributions by making it straightforward to add new features, models, or datasets.

### Why Modularity Matters

- **Ease of Use**: Modular components can be easily mixed and matched to suit specific needs, making the library versatile and adaptable.
- **Ease of Contribution**: With clear boundaries and interfaces between components, new contributors can easily understand where and how to add their contributions without having to navigate through a monolithic codebase.

### Contributing

We warmly welcome contributions of all forms:

- **Adding New Models or Features**: Enhance the library by integrating new machine learning models or introducing novel features.
- **Improving Existing Modules**: Contribute by refining the algorithms, optimizing performance, or enhancing the user experience.
- **Expanding Dataset Support**: Help by adding support for more datasets, improving data loading mechanisms, or contributing datasets.
- **Documentation and Examples**: Assist new users by improving documentation, adding tutorials, or providing example use cases.

Check out our [Contribution Guidelines](CONTRIBUTION.md) for more information on how to get started. Whether you're fixing a bug or proposing a new feature, your contributions are invaluable to making this project better for everyone.

## License

This project is licensed under the [MIT License](LICENSE). For more details, see the LICENSE file in the root directory.
