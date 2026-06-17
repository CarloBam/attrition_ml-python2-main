# Employee Attrition Prediction Project

This project aims to develop a machine learning application for predicting employee attrition using Streamlit. The application allows users to upload datasets, train models, and visualize results to gain insights into employee turnover.

## Project Structure

```
attrition-ml-streamlit
├── src
│   ├── app.py                # Main application file for the Streamlit app
│   ├── generate_sample.py     # Functions to generate synthetic datasets
│   ├── preprocessing.py       # Data cleaning and preprocessing functions
│   ├── features.py            # Feature engineering functions
│   ├── models.py              # Functions for training and evaluating models
│   ├── viz.py                 # Visualization functions for data and model performance
│   └── utils.py               # Utility functions for data loading and saving
├── data
│   └── sample_attrition_data.csv # Sample dataset for testing and demonstration
├── notebooks
│   └── generate_dataset.ipynb  # Jupyter notebook for dataset generation experimentation
├── tests
│   └── test_data_generation.py  # Unit tests for data generation functions
├── requirements.txt             # List of required Python packages
├── pyproject.toml               # Project configuration file
├── .gitignore                    # Files and directories to ignore by Git
└── README.md                     # Project documentation
```

## Installation

To set up the project, clone the repository and install the required packages:

```bash
git clone <repository-url>
cd attrition-ml-streamlit
pip install -r requirements.txt
```

## Usage

1. **Run the Streamlit App**: Start the application by running the following command:

   ```bash
   streamlit run src/app.py
   ```

2. **Upload Data**: Use the app interface to upload employee data in CSV format.

3. **Train Models**: Select features and train various machine learning models to predict attrition.

4. **Visualize Results**: Explore performance metrics, confusion matrices, and other visualizations to understand model performance.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.