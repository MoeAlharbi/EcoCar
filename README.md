# EcoCar

EcoCar is a machine learning project aimed at predicting vehicle fuel consumption and CO2 emissions based on key features like engine size, fuel type, and transmission. This project leverages multiple regression models to provide accurate predictions, helping to assess the environmental impact of vehicles.

---

## Features
- Exploratory Data Analysis (EDA) with visualizations for insights.
- Feature engineering, including creation and normalization of data.
- Regression models implemented:
  - Linear Regression
  - Random Forest Regressor
  - Support Vector Regressor (SVR)
- Evaluation metrics:
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R² Score
- Visualizations: Predicted vs. Actual plots.

---

## Installation Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/MoeAlharbi/EcoCar.git
cd EcoCar
```

### 2. Install Dependencies
Use the `requirements.txt` file to install all necessary Python libraries:
```bash
pip install -r requirements.txt
```

### 3. Download the Dataset
- Source: [Kaggle CO2 Emissions Dataset](https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles)
- Save the dataset as `CO2_Emissions_Canada.csv` and place it in the `/data` folder.

### 4. Run the Script
To run the project and view results:
```bash
python project_script.py
```

---

## Project Structure
```
EcoCar/
│
├── EcoCar.py                  # Main script containing all project code
├── README.md                  # Project documentation
├── requirements.txt           # Dependencies for the project
├── /data                      # Folder for dataset
   └── CO2_Emissions_Canada.csv
```

---

## Usage

1. **Preprocess the Data**:
   - The `engineer_features()` function combines and normalizes features for model training.

2. **Perform EDA**:
   - The `perform_eda()` function generates insightful visualizations (e.g., distribution plots, correlation heatmaps).

3. **Train Models**:
   - Use the `train_models()` function to train and evaluate Linear Regression, Random Forest, and SVR models.

4. **Evaluate Models**:
   - The `evaluate_models()` function visualizes predicted vs. actual performance and outputs key metrics (e.g., RMSE, R²).

---

## Results

### Model Performance
| Model                     | Test MSE | Test RMSE | R² Score |
|---------------------------|----------|-----------|----------|
| Linear Regression         | 0.0104   | 0.1019    | 0.9897   |
| Random Forest Regressor   | 0.0057   | 0.0757    | 0.9943   |
| Support Vector Regressor  | 0.0057   | 0.0752    | 0.9944   |

### Visualizations
1. **Prediction vs. Actual**:
   - Scatterplots showing the relationship between predicted and actual CO2 emissions.
2. **Correlation Heatmap**:
   - Highlights the relationships between features like engine size, fuel consumption, and CO2 emissions.

---

## Limitations
- Dataset is limited to vehicles in Canada and may not generalize globally.
- High cardinality in some features (e.g., `Make`) might affect model performance.
- Linear Regression performs slightly worse due to its inability to capture non-linear relationships.

---

## Future Improvements
1. Expand the dataset to include vehicles from different regions and years.
2. Use advanced models such as Gradient Boosting Machines (XGBoost, LightGBM).
3. Build a web-based interface to allow users to input vehicle specifications and get CO2 predictions.

---

## Acknowledgments
- Dataset provided by [Kaggle](https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles).
- Project inspired by the goal of sustainable transportation and reducing vehicle emissions.


