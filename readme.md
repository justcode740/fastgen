# Blazing Fast, Generic Library for Automated Feature and Model Selection

This library is designed to offer a blazing-fast, generic solution for automated feature and model selection, tailored for large datasets. Leveraging the power of rust parallelism and abstraction, alongside advanced model selection techniques, it streamlines the process of preparing data and choosing the optimal machine learning model with best feature sets for your needs.

## Key Features

### Data Parallelism and Dataset Abstraction
- Offers a **unified interface** to interact with various datasets, whether they are part of `smartcore::dataset`, stored locally, or distributed across a network. This abstraction simplifies data handling and ensures efficient processing across different storage solutions.

### Model Parallelism
- Implements model parallelism to distribute the training of complex models across multiple computing nodes, significantly reducing training times and facilitating the handling of larger, more intricate models.

### Parallelism and Pipelining Across Feature Selection Stages
- Enhances the efficiency of the feature selection process through parallelism and pipelining, ensuring rapid identification of the most relevant features for your models.

## Roadmap (Todo)
- **Data Loading**: Enable actual loading of datasets from both network sources and local storage, likely involving storage links for seamless data integration.
- **Model Generality**: Refactor the current system to make the model handling truly generic, adding support for more machine learning models from various libraries.
- **Model Classification and Refactoring**: Analyze and classify different models based on their performance and required refinements, focusing on metrics-wise evaluation for better decision-making.
- **Advanced Genetic Features**: Integrate more sophisticated genetic algorithm features, such as elitism, to enhance the selection process's effectiveness and efficiency.
- **User Interface Enhancements**: Develop a more user-friendly CLI and output system, making the library more accessible and easier to use for a broader audience.
- **Frontend Visualization**: Consider the creation of a frontend interface to visualize the generic selection process, offering users insight into the decision-making process behind feature and model selection.

## Getting Involved
Contributions to this project are highly welcome. Whether you're looking to fix bugs, propose new features, or enhance the documentation, your input is invaluable. Check out our issues list for areas where you can help, or feel free to suggest new ideas for the project.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
