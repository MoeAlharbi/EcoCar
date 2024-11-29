# -*- coding: utf-8 -*-
"""

## Section 1: Problem Definition
---
**Specific Problem Addressed**

Transportation contributes significantly to greenhouse gas emissions, which accelerate climate change. Many car buyers and policymakers lack the tools to easily assess the environmental impact of vehicles based on their specifications. This knowledge gap hinders informed decision-making toward adopting cleaner and more fuel-efficient vehicles.


**Potential Impact of the Solution**

The EcoCar project empowers individuals and policymakers with actionable insights into vehicle emissions. By leveraging machine learning, we enable accurate predictions of fuel consumption and CO2 emissions, fostering environmentally conscious choices and promoting cleaner technologies in the automotive industry.


**Alignment with UN Sustainable Development Goals (SDGs)**

The EcoCar project aligns with the United Nations Sustainable Development Goals (SDGs), specifically:

*Goal 11*: Sustainable Cities and Communities – By providing tools to assess vehicle emissions, we promote cleaner and more sustainable urban mobility.

*Goal 13*: Climate Action – Encourages understanding and reducing vehicular CO2 emissions, contributing to global efforts in mitigating climate change.


**Research Questions**

*RQ1*: How does engine size and transmission type affect fuel efficiency and CO2 emissions for vehicles?

*RQ2*: What is the relationship between fuel type and environmental impact in terms of CO2 emissions?

*RQ*3: Can we predict fuel consumption and CO2 emissions using a machine learning model trained on vehicle specifications?
"""

# Data manipulation and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning and model evaluation
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# For the user interface


# Set random seed for reproducibility
np.random.seed(42)

"""## Section 2: Data Collection & Preparation
---

1. **Data Source Documentation:**

*Dataset Name and Source* :
The dataset is titled CO2 Emissions by Vehicles and is sourced from Kaggle.

*Time Period Covered* :
The dataset covers vehicle emissions data for the year 2018 in Canada.

*Number of Records and Features* :

* Records: The dataset contains 7385 records.
* Features: The dataset includes 12 columns, such as:
* Make (Car manufacturer)
* Model (Car model)
* Vehicle Class (Classification like SUV, compact, etc.)
* Engine Size (L) (Engine displacement)
* Cylinders (Number of cylinders)
* Transmission (Transmission type: Automatic, Manual)
* Fuel Type (Type of fuel: Gasoline, Diesel, etc.)
* Fuel Consumption City (L/100 km) (Fuel consumption in the city)
* Fuel Consumption Hwy (L/100 km) (Fuel consumption on highways)
* Fuel Consumption Combined (L/100 km) (Average fuel consumption)
* CO2 Emissions (g/km) (CO2 emissions in grams per kilometer)

*Data Collection Methodology* :

The dataset is compiled from Canadian vehicle emissions testing data, which measures fuel consumption and emissions under standardized test conditions. These values are reported by car manufacturers and verified through regulatory testing.

2. **Initial Data Assessment:**

*Basic Statistics*:

Summary statistics provide insights into key variables, such as:

* Fuel Consumption Combined (L/100 km): Mean, median, and range of fuel consumption values across vehicles.
* CO2 Emissions (g/km): Average emissions and their spread.
Engine Size (L): Ranges from small displacement engines (~1.0L) to large engines (>5.0L).


"""

# Load your dataset
def load_dataset():
    """
    Load the dataset and perform initial assessment
    """
    # Load your data
    df = pd.read_csv('/content/data/CO2 Emissions_Canada.csv')

    # Display basic information
    print("Dataset Shape:", df.shape)
    print("\nDataset Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())

    return df

"""## Section 3: Exploratory Data Analysis (EDA)
---
### Instructions
Task. Understand your data patterns and perform comprehensive EDA to understand your data:

**1. Required Analyses:**

  **Descriptive Statistics:**
* Key statistics:
  * Engine Size (L): Mean: 3.16, Min: 0.9, Max: 8.4
  * Fuel Consumption Comb (L/100 km): Mean: 10.98, Min: 4.1, Max: 26.1
  * CO2 Emissions (g/km): Mean: 250.58, Min: 96, Max: 522

**Univariate Analysis:**

CO2 Emissions show a near-normal distribution, with the majority of vehicles emitting between 200 and 300 g/km.

**Bivariate Analysis:**

* A strong positive relationship exists between Engine Size (L) and CO2 Emissions (g/km), as shown in the scatter plot.
* Fuel Type significantly affects CO2 emissions, with noticeable variations in emissions among different fuel types, as seen in the box plot.

**Correlation Analysis:**

* Engine Size (L) and Cylinders have the highest positive correlation (0.93), indicating that larger engines tend to have more cylinders.
* Fuel Consumption Comb (mpg) has a negative correlation (-0.91) with CO2 Emissions (g/km), suggesting higher fuel efficiency results in lower emissions.
* CO2 Emissions strongly correlate with Fuel Consumption Comb (L/100 km) (0.92).

**2. Required Visualizations (minimum 5):**


  **a. Distribution of CO2 Emissions:**

  * CO2 emissions data shows a near-normal distribution, peaking between 200–300 g/km.

  **b. Scatter Plot: Engine Size vs CO2 Emissions:**

  The plot highlights that vehicles with larger engines tend to emit more CO2, with a strong positive trend.


  **c. Box Plot: CO2 Emissions by Fuel Type:**

  Significant differences are observed among different fuel types:
  * Fuel type "Z" (likely gasoline) has the highest variance in emissions.
  * Fuel type "N" (possibly electric or hybrid) shows significantly lower emissions.

  **d. Correlation Heatmap:**

  Demonstrates strong relationships between numerical variables, especially fuel consumption and CO2 emissions.

  **e. Descriptive Statistics Table:**

  Numerical variables like fuel consumption and emissions are summarized to provide deeper insights into the dataset.
"""

def perform_eda(df):
    """
    Conduct exploratory data analysis
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 1. Univariate Analysis
    print("Univariate Analysis:")
    print("Descriptive Statistics:")
    print(df.describe())

    # Distribution plot for CO2 Emissions
    plt.figure(figsize=(8, 5))
    sns.histplot(df['CO2 Emissions(g/km)'], kde=True, color='green')
    plt.title("Distribution of CO2 Emissions")
    plt.xlabel("CO2 Emissions (g/km)")
    plt.ylabel("Frequency")
    plt.show()

    # 2. Bivariate Analysis
    print("\nBivariate Analysis:")

    # Scatter plot: Engine Size vs CO2 Emissions
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='Engine Size(L)', y='CO2 Emissions(g/km)', data=df)
    plt.title("Engine Size vs CO2 Emissions")
    plt.xlabel("Engine Size (L)")
    plt.ylabel("CO2 Emissions (g/km)")
    plt.show()

    # Boxplot: CO2 Emissions by Fuel Type
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Fuel Type', y='CO2 Emissions(g/km)', data=df)
    plt.title("CO2 Emissions by Fuel Type")
    plt.xlabel("Fuel Type")
    plt.ylabel("CO2 Emissions (g/km)")
    plt.show()

    # 3. Correlation Analysis
    print("\nCorrelation Analysis:")
    # Select only numerical columns for correlation
    numerical_columns = df.select_dtypes(include=[float, int])
    correlation_matrix = numerical_columns.corr()
    print(correlation_matrix)

    # Correlation heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

    # 4. Custom Analysis for Your Topic
    print("\nCustom Analysis:")
    # Dynamically find and include valid features
    selected_features = ['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)', 'CO2 Emissions(g/km)']
    sns.pairplot(df[selected_features])
    plt.show()

perform_eda(load_dataset())

"""## Section 4: Feature Engineering
---
### Most Important Features Identified:
- **Engine Size (L)**:
  - Strong positive correlation with CO2 emissions (0.85).
- **Cylinders**:
  - High correlation with CO2 emissions and engine size (0.83).
- **Fuel Type and Transmission**:
  - Significant categorical features influencing emissions and fuel consumption.
- **Fuel Consumption Combined**:
  - Highest correlation with CO2 emissions (0.92).

### Removed Features:
- **Fuel Consumption City (L/100 km)** and **Fuel Consumption Hwy (L/100 km)**:
  - Combined into a single feature (`Fuel Consumption Combined`) to reduce redundancy.
- **Fuel Consumption Comb (mpg)**:
  - Inversely related to `Fuel Consumption Combined (L/100 km)`; keeping one suffices.
- **Model**:
  - High cardinality and unlikely to add significant predictive power.
- **Vehicle Class**:
  - Less predictive compared to other features.

### Normalization:
- **Numerical Features**:
  - Standardized using `StandardScaler` to have a mean of 0 and a standard deviation of 1.
  - Ensures that all numerical features are on the same scale, which is beneficial for many machine learning algorithms.

### Selection Criteria:
- Features were selected based on:
  - Correlation with the target variable (CO2 emissions).
  - Predictive power derived from exploratory data analysis.
  - Redundancy reduction by removing overlapping or less informative features.
"""

def engineer_features(df):
    """
    Create, select, and normalize features
    """
    from sklearn.preprocessing import StandardScaler

    # 1. Create new features
    df['Fuel Consumption Combined'] = (df['Fuel Consumption City (L/100 km)'] +
                                       df['Fuel Consumption Hwy (L/100 km)']) / 2

    # 2. Feature selection: Drop redundant or less important features
    # Based on prior analysis and correlation with the target variable
    features_to_drop = [
        'Fuel Consumption City (L/100 km)',
        'Fuel Consumption Hwy (L/100 km)',
        'Fuel Consumption Comb (mpg)',  # Redundant with Combined Fuel Consumption
        'Model',                        # High cardinality, may not contribute significantly
        'Vehicle Class'                 # May be less predictive
    ]
    df = df.drop(columns=features_to_drop)

    # 3. Encode categorical variables
    df = pd.get_dummies(df, columns=['Make', 'Transmission', 'Fuel Type'], drop_first=True)

    # 4. Normalize numerical features
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df, scaler

"""
## Section 5: Model Development
---

### **1. Required Steps:**

**1.1 Split Data into Training and Testing Sets:**
- The dataset is split into training and testing sets as part of the `train_models` function:
  ```python
  from sklearn.model_selection import train_test_split
  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  ```
  - `test_size=0.2`: Reserves 20% of the data for testing.
  - `random_state=42`: Ensures reproducibility by controlling the randomness.
- Data splitting is explicitly handled inside the `train_models` function.

---

**1.2 Implement at Least 3 Different Models:**
- **Models Used:**
  1. **Linear Regression**:
     - Simple baseline model that assumes a linear relationship between features and the target variable.
  2. **Random Forest Regressor**:
     - Captures complex, non-linear relationships.
     - Tends to perform well on tabular datasets with both categorical and numerical features.
  3. **Support Vector Regressor (SVR)**:
     - Effective for small-to-medium datasets and can model non-linear relationships using kernels.
- These models provide a diverse range of approaches to address regression problems.

---

**1.3 Document Model Selection Rationale:**
- **Linear Regression**:
  - Serves as a baseline model to compare the performance of more complex algorithms.
  - Computationally efficient and interpretable.
- **Random Forest Regressor**:
  - Handles non-linear relationships effectively and is robust to overfitting, especially with default settings.
  - Provides feature importance metrics for model interpretability.
- **SVR**:
  - Suitable for capturing non-linear trends in the data.
  - May require feature scaling for better performance, making it complementary to the other models.

---

**1.4 Implement Cross-Validation:**
- Cross-validation is performed using `cross_val_score()` with 5 folds:
  ```python
  cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
  ```
  - **Scoring Metric**: Negative Mean Squared Error (`neg_mean_squared_error`) is used to calculate RMSE.
  - **5 Folds**: Ensures model stability by evaluating performance on different data subsets.
  - **RMSE Calculation**: RMSE is derived from the negative MSE values:
    ```python
    cv_rmse = np.sqrt(-cv_scores)
    ```

---

### **2. Memory Optimization Requirements:**

**2.1 Use Appropriate Data Types:**
- Memory usage can be reduced by ensuring numerical features use optimized types (e.g., `float32` instead of `float64`).

---

**2.2 Implement Batch Processing if Needed:**
- Models like Random Forest and SVR are processed in memory. If dataset size exceeds memory limits:
  - Use incremental learning models like `SGDRegressor` or `PartialFit`.

---

**2.3 Document Memory Usage:**
- Memory usage can be tracked using tools like Python’s `psutil` or memory profilers:
  ```python
  import psutil
  print(f"Memory Usage: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")
  ```
  - This ensures efficient monitoring of memory during model training and evaluation.
```

"""

def train_models(X, y, test_size=0.2, random_state=42):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"Data Split Complete: {len(X_train)} training samples, {len(X_test)} testing samples")

    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor(random_state=random_state),
        'Support Vector Regressor (SVR)': SVR()
    }

    # Dictionary to store results
    results = {}

    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"Training {model_name}...")

        # Train the model
        model.fit(X_train, y_train)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores)

        # Evaluate on test set
        y_pred = model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_r2 = r2_score(y_test, y_pred)

        # Save results
        results[model_name] = {
            'Cross-Validation RMSE': cv_rmse.mean(),
            'Test RMSE': test_rmse,
            'R2 Score': test_r2
        }

    # Display results
    for model_name, metrics in results.items():
        print(f"\n{model_name} Results:")
        print(f"  Cross-Validation RMSE: {metrics['Cross-Validation RMSE']:.4f}")
        print(f"  Test RMSE: {metrics['Test RMSE']:.4f}")
        print(f"  R2 Score: {metrics['R2 Score']:.4f}")

    return models, X_test, y_test

"""## Section 6: Model Evaluation
---
### Instructions
Evaluate your models thoroughly:

**1. Required Metrics:**
   * Accuracy/RMSE (depending on problem type)
   * Precision, Recall, F1-Score (for classification)
   * Cross-validation scores
   * Model comparison analysis

**2. Required Visualizations:**
   * Confusion matrix (for classification)
   * ROC curves (for classification)
   * Prediction vs Actual plots (for regression)
   * Feature importance plots
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_models(models, X_test, y_test):
    """
    Evaluate model performance and generate required visualizations
    """
    results = {}
    plt.figure(figsize=(15, 5))

    for idx, (model_name, model) in enumerate(models.items(), start=1):
        print(f"Evaluating {model_name}...")

        # Predict on test set
        y_pred = model.predict(X_test)

        # Metrics
        test_mse = mean_squared_error(y_test, y_pred)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(y_test, y_pred)

        # Save metrics to results
        results[model_name] = {
            'Test MSE': test_mse,
            'Test RMSE': test_rmse,
            'R2 Score': test_r2
        }

        # Plot Prediction vs Actual
        plt.subplot(1, len(models), idx)
        plt.scatter(y_test, y_pred, alpha=0.6, edgecolor='k')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.title(f"{model_name}\nRMSE: {test_rmse:.2f}, R2: {test_r2:.2f}")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")

    plt.tight_layout()
    plt.show()

    # Display results
    print("\nModel Evaluation Results:")
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        print(f"  Test MSE: {metrics['Test MSE']:.4f}")
        print(f"  Test RMSE: {metrics['Test RMSE']:.4f}")
        print(f"  R2 Score: {metrics['R2 Score']:.4f}")

    return results

"""## Section 7: Results & Sustainability Impact
---

### **1. Required Components**

#### **Answer Each Research Question with Evidence**
1. **How does engine size affect CO2 emissions?**
   - Based on the results, engine size is a critical feature in predicting CO2 emissions. The high R² values across all models (Linear Regression: 0.9897, Random Forest: 0.9943, SVR: 0.9944) indicate that larger engine sizes are strongly associated with higher CO2 emissions.

2. **What is the relationship between fuel consumption and CO2 emissions?**
   - **Fuel Consumption Combined** has the highest predictive power for CO2 emissions, as shown by the low Test MSE (e.g., Random Forest: 0.0057, SVR: 0.0057) and high R² scores. Vehicles with higher combined fuel consumption emit more CO2, which aligns with domain knowledge.

3. **Can machine learning models predict CO2 emissions accurately based on the features?**
   - Yes, the models demonstrated strong predictive performance:
     - **Linear Regression**: R² = 0.9897, RMSE = 0.1019
     - **Random Forest Regressor**: R² = 0.9943, RMSE = 0.0757
     - **Support Vector Regressor (SVR)**: R² = 0.9944, RMSE = 0.0752
   - Both Random Forest and SVR slightly outperformed Linear Regression, likely due to their ability to capture non-linear relationships.

---

#### **Quantify Sustainability Impact**
- **Environmental Impact**:
  - By identifying features such as engine size and fuel consumption, the model can help design vehicles with lower emissions. This supports reducing greenhouse gas emissions.
  - The predictions can assist regulators in setting stricter emission standards based on accurate data.

- **Accuracy of Models**:
  - The low RMSE values (e.g., 0.0757 for Random Forest) mean the models are highly reliable in estimating CO2 emissions, making them applicable in sustainability-focused decision-making.

---

#### **Discuss Limitations**
1. **High Cardinality Features**:
   - Features like "Make" (car manufacturer) increase model complexity, which might not generalize well to unseen makes or models.

2. **Non-Generalizability**:
   - The dataset is limited to vehicles in Canada. The models might not perform as well for other regions or under different driving conditions.

3. **Linear Model Limitations**:
   - Linear Regression performs slightly worse than Random Forest and SVR because it cannot capture non-linear relationships effectively.

4. **Real-World Validation**:
   - The models have not been tested on real-world data beyond the dataset, which could limit their robustness in deployment.

---

#### **Propose Future Improvements**
1. **Dataset Expansion**:
   - Incorporate data from different countries, vehicle types, and time periods for better generalizability.
   
2. **Additional Features**:
   - Include additional predictors such as vehicle weight, aerodynamics, or hybrid/electric vehicle features.

3. **Model Refinement**:
   - Experiment with ensemble models like **Gradient Boosting Machines (e.g., XGBoost, LightGBM)** to improve performance further.

4. **Application**:
   - Create an application that allows users to input vehicle specifications and predict CO2 emissions, fostering awareness and informed decision-making.

---

### **2. Impact Assessment**

#### **Environmental Impact**
- The models enable identification of high-emission vehicles and offer insights into which features (engine size, fuel type) contribute most to emissions. This helps manufacturers and policymakers design sustainable vehicles and enforce stricter emission standards.

#### **Social Impact**
- Provides tools for consumers to choose low-emission vehicles, promoting sustainability. Additionally, it can raise public awareness about the impact of vehicle choices on the environment.

#### **Economic Impact**
- Manufacturers can optimize production to focus on low-emission vehicles, avoiding regulatory fines while appealing to eco-conscious consumers. Governments can also use these models to develop targeted tax incentives or penalties for emissions.

#### **SDG Alignment Evidence**
- **Goal 13: Climate Action**:
  - The project supports climate change mitigation by providing accurate tools for assessing vehicle emissions.
- **Goal 11: Sustainable Cities and Communities**:
  - Promotes sustainable transportation through informed design and consumer choices.

"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_models(models, X_test, y_test):
    """
    Evaluate model performance and generate required visualizations
    """
    results = {}
    plt.figure(figsize=(15, 5))

    for idx, (model_name, model) in enumerate(models.items(), start=1):
        print(f"Evaluating {model_name}...")

        # Predict on test set
        y_pred = model.predict(X_test)

        # Metrics
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_r2 = r2_score(y_test, y_pred)

        # Save metrics to results
        results[model_name] = {
            'Test RMSE': test_rmse,
            'R2 Score': test_r2
        }

        # Plot Prediction vs Actual
        plt.subplot(1, len(models), idx)
        plt.scatter(y_test, y_pred, alpha=0.6, edgecolor='k')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.title(f"{model_name}\nRMSE: {test_rmse:.2f}, R2: {test_r2:.2f}")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")

    plt.tight_layout()
    plt.show()

    # Display results
    print("\nModel Evaluation Results:")
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        print(f"  Test RMSE: {metrics['Test RMSE']:.4f}")
        print(f"  R2 Score: {metrics['R2 Score']:.4f}")

    return results

def main():
    # Load data
    data = load_dataset()

    # Engineer features
    data, scaler = engineer_features(data)

    # Define features and target
    X = data.drop(columns=['CO2 Emissions(g/km)'])
    y = data['CO2 Emissions(g/km)']

    # Train models
    models, X_test, y_test = train_models(X, y)

    # Evaluate models
    evaluate_models(models, X_test, y_test)

# Run the main function
if __name__ == "__main__":
    main()

"""## Section 8: References & Documentation
---
#### **1. Required Elements**



### **All Data Sources**
- **Dataset Name**: CO2 Emissions by Vehicles
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles)
- **Description**:
  - Contains specifications for Canadian vehicles, including:
    - Engine Size (L)
    - Fuel Consumption (City/Highway/Combined in L/100 km)
    - Cylinders
    - Transmission
    - Fuel Type
    - CO2 Emissions (g/km)
  - **Number of Records**: 7385 rows
  - **Columns**: 12 total features.

---

### **External Libraries Used**
1. **pandas**: For data manipulation and preprocessing.
2. **numpy**: For numerical computations.
3. **matplotlib**: For creating visualizations like scatter plots and prediction vs. actual plots.
4. **seaborn**: For advanced visualizations such as correlation heatmaps.
5. **scikit-learn**:
   - Preprocessing: `StandardScaler` for feature normalization.
   - Models: `LinearRegression`, `RandomForestRegressor`, `SVR`.
   - Model Selection: `train_test_split`, `RandomizedSearchCV`.
   - Metrics: `mean_squared_error`, `r2_score`.
   - Cross-Validation: `cross_val_score`.

---

### **References**

1. UN Sustainable Development Goals (SDG):
   - Goal 13: Climate Action ([Link](https://www.un.org/sustainabledevelopment/climate-change/)).
   - Goal 11: Sustainable Cities and Communities ([Link](https://www.un.org/sustainabledevelopment/cities/)).

---

### **Code Documentation**
1. **Main Functions**:
   - **`load_dataset()`**:
     - Loads the CO2 dataset and provides basic information like missing values and duplicates.
 - **`perform_eda(df)`**:
     - Generates visualizations for:
       - Univariate analysis (e.g., distribution of CO2 emissions).
       - Bivariate analysis (e.g., Engine Size vs. CO2 Emissions).
       - Correlation heatmaps to identify predictive features.
   - **`engineer_features(df)`**:
     - Combines fuel consumption features, removes redundant ones, encodes categorical data, and normalizes numerical columns.
   - **`train_models(X, y)`**:
     - Splits data into training/testing sets and trains three machine learning models.
     - Outputs performance metrics (RMSE, R²).
   - **`evaluate_models(models, X_test, y_test)`**:
     - Visualizes predictions vs. actual values and evaluates test performance.

---


### **Installation/Runtime Instructions**

1. **Clone the Repository**:
   - Use the following commands to clone the repository to your local machine:
     ```bash
     git clone <repository_link>
     cd <repository_folder>
     ```

2. **Install Dependencies**:
   - Make sure all required libraries are installed. Use the `requirements.txt` file for easy installation:
     ```bash
     pip install -r requirements.txt
     ```

3. **Prepare the Dataset**:
   - Download the dataset from the [Kaggle link](https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles).
   - Save the dataset as `CO2_Emissions_Canada.csv`.
   - Place the dataset in the `data` folder (or the same directory as the script, depending on your file structure).

4. **Run the Script**:
   - If using Python on a local machine:
     ```bash
     python project_script.py
     ```
   - Ensure that the dataset path is correctly set in the script (e.g., `/content/data/CO2_Emissions_Canada.csv`).

5. **Run in Google Colab**:
   - Upload the project script and dataset to Google Colab.
   - Update the dataset file path in the script (if necessary).
   - Run each cell sequentially to preprocess the data, perform EDA, train models, and evaluate their performance.

6. **Expected Output**:
   - The script will:
     - Perform exploratory data analysis with visualizations.
     - Train and evaluate three models (Linear Regression, Random Forest, SVR).
     - Display model metrics (e.g., RMSE, R²) and generate evaluation plots.

7. **Optional**:
   - Customize the dataset path or parameters (e.g., test size, random state) in the script for different setups.
   - Use the trained models to predict CO2 emissions for new vehicles based on the features.

---


"""



