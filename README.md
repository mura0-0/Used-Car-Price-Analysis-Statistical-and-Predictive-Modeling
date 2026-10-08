# Used Car Price Analysis: Statistical and Predictive Modeling

## Abstract
This project analyzes the UCI Automobile Dataset to identify the key factors that influence used car prices and to build a predictive model. After performing data cleaning, exploratory data analysis, and statistical testing (Pearson correlation and ANOVA), we developed Ridge regression models with polynomial features. 

## Introduction
Predicting used car prices is a practical and commercially relevant problem for buyers, sellers, and online marketplaces. Accurate price estimation helps reduce information asymmetry and supports better decision-making.

**Research questions:**
- Which features have the strongest influence on used car prices?
- Can we build a reliable predictive model using statistical and machine learning techniques?

## Data
**Source:** [UCI Machine Learning Repository – Automobile Dataset](https://archive.ics.uci.edu/ml/datasets/automobile)

The dataset contains **205 records** and **26 attributes**, including:
- Quantitative variables (engine-size, horsepower, curb-weight, etc.)
- Categorical variables (make, body-style, drive-wheels, fuel-type, etc.)
- Target variable: `price`

### Data Cleaning Summary
- Missing values were handled by mean imputation for continuous variables and mode imputation for categorical variables
- Categorical variables such as `num-of-cylinders` and `num-of-doors` were converted to numeric form
- One-hot encoding was applied to selected categorical features

## Methodology

### Exploratory Data Analysis
- Box plots to examine price distributions across categorical variables
- Correlation heatmap and Pearson correlation analysis

### Statistical Tests
- **Pearson correlation** to measure linear relationships between numeric features and price
- **ANOVA** to test whether categorical variables significantly affect price

### Modeling Approach
Ridge Regression (to control overfitting)

The model was implemented using `scikit-learn` Pipelines that include:
- PolynomialFeatures
- Ridge Regression
- Hyperparameter tuning with GridSearchCV

### Evaluation Metrics
- R² (Coefficient of Determination)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- 5-Fold Cross-Validation

## Results

### Key Findings
- `engine-size`, `horsepower`, `curb-weight`, and `highway-mpg` showed the strongest relationships with price
- Several categorical variables (e.g., `drive-wheels`) were statistically significant according to ANOVA

## Discussion / Limitations

**What worked well:**
- Feature selection based on statistical evidence improved model clarity and performance
- Ridge regression effectively controlled overfitting when using polynomial terms

**Limitations:**
- The dataset is relatively small (205 rows)
- Some categorical groups (e.g., rear engine location) have very few observations
- Missing values required imputation, which introduces uncertainty

**Future improvements:**
- Collect a larger and more recent dataset
- Experiment with other algorithms 

## How to Reproduce

### Installation
```bash
pip install -r requirements.txt