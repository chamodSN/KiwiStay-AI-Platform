# Airbnb Price Predictor NZ

## Overview
This project builds a machine learning pipeline to predict Airbnb listing prices in New Zealand (NZ) using regression models. The dataset (`data/listings.csv`) contains features like location (latitude, longitude), room type, availability, and minimum nights. The pipeline includes data preprocessing, feature engineering, regression modeling (including Decision Tree, Random Forest, etc.), ensemble methods, and a Streamlit web app for interactive predictions. 

The target variable is `log_price` (log-transformed price) to handle skewness, with predictions converted back to NZD.

**Key goals:**
- Clean and transform raw data for modeling.
- Train and optimize multiple regression models.
- Combine models in ensembles for better accuracy.
- Deploy predictions via a user-friendly Streamlit app.

---

## Project Structure
├── **common/**  
│   └── `config.py` # Constants: RANDOM_SEED, INPUT_CSV  
├── **data/**  
│   ├── `listings.csv` # Raw and processed Airbnb dataset  
│   ├── **models/** # Saved models and parameters  
│   │   ├── `decision_tree_basic.pkl`  
│   │   ├── `decision_tree_basic_params.pkl`  
│   │   ├── `decision_tree_optimized.pkl`  
│   │   ├── `decision_tree_optimized_params.pkl`  
│   │   ├── `gradient_boosting_basic.pkl`  
│   │   ├── `gradient_boosting_basic_params.pkl`  
│   │   ├── `gradient_boosting_optimized.pkl`  
│   │   ├── `gradient_boosting_optimized_params.pkl`  
│   │   ├── `linear_regression_basic.pkl`  
│   │   ├── `linear_regression_basic_params.pkl`  
│   │   ├── `linear_regression_optimized.pkl`  
│   │   ├── `linear_regression_optimized_params.pkl`  
│   │   ├── `random_forest_basic.pkl`  
│   │   ├── `random_forest_basic_params.pkl`  
│   │   ├── `random_forest_optimized.pkl`  
│   │   ├── `random_forest_optimized_params.pkl`  
│   │   ├── `xgboost_basic.pkl`  
│   │   ├── `xgboost_basic_params.pkl`  
│   │   ├── `xgboost_optimized.pkl`  
│   │   ├── `xgboost_optimized_params.pkl`  
│   │   ├── `adaboost_ensemble.pkl`  
│   │   ├── `adaboost_ensemble_params.pkl`  
│   │   ├── `stacking_ensemble.pkl`  
│   │   └── `bagging_ensemble.pkl`  
│   └── **processed/** # Train/test splits  
│       ├── `X_train.pkl`  
│       ├── `X_test.pkl`  
│       ├── `y_train.pkl`  
│       └── `y_test.pkl`  
├── **preprocessing/**  
│   ├── `data_cleaning.py`  
│   ├── `data_discritization.py`  
│   ├── `data_quality.py`  
│   ├── `data_reduction.py`  
│   ├── `data_transform.py`  
│   └── `feature_importance.py`  
├── **regression/**  
│   ├── **decision_tree_regressor/**  
│   │   ├── `basic_parameter.py`  
│   │   └── `optimized_parameter.py`  
│   ├── **gradient_boosting_regressor/**  
│   │   ├── `basic_parameter.py`  
│   │   └── `optimized_parameter.py`  
│   ├── **linear_regression/**  
│   │   ├── `basic_parameter.py`  
│   │   └── `optimized_parameter.py`  
│   ├── **random_forest_regressor/**  
│   │   ├── `basic_parameters.py`  
│   │   └── `optimized_parameter.py`  
│   ├── **xgboost_regressor/**  
│   │   ├── `basic_parameter.py`  
│   │   └── `optimized_parameter.py`  
│   ├── **ensemble/**  
│   │   ├── `adaboost_ensemble.py`  
│   │   ├── `bagging_ensemble.py`  
│   │   └── `stacking_ensemble.py`  
│   └── `report_generator.py`  
├── **frontend/**  
│   └── `streamlit_app_frontend.py`  
└── `README.md`


---

## Setup Instructions

### Clone Repository
```bash
git clone <repository_url>
cd airbnb-price-predictor-nz
```
### Install Dependencies

Ensure Python 3.8+ is installed. Then run:
```
pip install pandas numpy scikit-learn xgboost joblib streamlit matplotlib seaborn category_encoders
```
### Prepare Data

Place listings.csv in data/ (expected columns: id, host_id, neighbourhood, room_type, price, etc.).

### Run Pipeline

Execute scripts in order:
```
python preprocessing/data_quality.py
python preprocessing/data_cleaning.py
python preprocessing/data_discritization.py
python preprocessing/data_reduction.py
python preprocessing/data_transform.py
python preprocessing/feature_importance.py
python regression/linear_regression/basic_parameter.py
python regression/linear_regression/optimized_parameter.py
python regression/decision_tree_regressor/basic_parameter.py
python regression/decision_tree_regressor/optimized_parameter.py
python regression/random_forest_regressor/basic_parameters.py
python regression/random_forest_regressor/optimized_parameter.py
python regression/gradient_boosting_regressor/basic_parameter.py
python regression/gradient_boosting_regressor/optimized_parameter.py
python regression/xgboost_regressor/basic_parameter.py
python regression/xgboost_regressor/optimized_parameter.py
python regression/ensemble/adaboost_ensemble.py
python regression/ensemble/stacking_ensemble.py
python regression/ensemble/bagging_ensemble.py
python regression/report_generator.py
```
Run Streamlit App
```
streamlit run frontend/streamlit_app_frontend.py
```
### Usage

**Preprocessing**: Run `data_quality.py` → `data_cleaning.py` → `data_discritization.py` → `data_reduction.py` → `data_transform.py` → `feature_importance.py`.

**Modeling**: Run basic and optimized scripts for each model. Outputs saved in `data/models/`.

**Reporting**: Run `report_generator.py` to compare R² scores.

**App**: Use the Streamlit app to select a model and input listing details to predict price in NZD.

### Component Details

#### Common
- `config.py`: Defines `RANDOM_SEED=42` and `INPUT_CSV="data/listings.csv"`.

#### Preprocessing
- `data_quality.py`: Checks missing values, duplicates, types, distributions.
- `data_cleaning.py`: Handles missing values, outliers, and adds features.
- `data_discritization.py`: Bins price and availability for analysis.
- `data_reduction.py`: Drops irrelevant columns and applies PCA.
- `data_transform.py`: Log-transforms price, encodes categories, scales, and adds polynomial interactions.
- `feature_importance.py`: Computes correlations, mutual info, RF feature importance, and saves train/test splits.

#### Regression
- **Linear Regression**: Baseline (R²~0.30). Optimized tweaks `fit_intercept`/`positive`.
- **Decision Tree**: Basic R²~0.37; optimized improves generalization.
- **Random Forest**: Ensemble of trees, R²~0.49; optimized tunes `n_estimators`/`max_depth`.
- **Gradient Boosting**: Sequential boosting, R²~0.48; optimized tunes hyperparameters.
- **XGBoost**: Fast boosting, R²~0.49; optimized similar to Gradient Boosting.
- **Ensemble Models**: AdaBoost, Bagging, Stacking combine optimized models for better accuracy.
- **Report Generator**: Summarizes R² scores and insights.

#### Frontend
- Streamlit app allows users to select a model, input features, and predict price in NZD. Feature importance is displayed for tree-based models.

### Results
- **Best Models**: Random Forest, XGBoost (R²~0.49)
- **Decision Tree**: Moderate (R²~0.37), used in ensembles
- **Insights**: Non-linear patterns favor tree-based models. Missing features (amenities, seasonality) limit R² < 0.5.

### Future Improvements
- Add features: amenities, seasonality, review text analysis
- Expand hyperparameter tuning
- Deploy app on cloud (e.g., Streamlit Cloud)
- Fix bugs in transform/reduction scripts

### Contributors
- **Chamod**: Data Preprocessing
- **Sandun**: 
- **Monali**: Data Preprocessing
- **Lihini**: Streamlit Frontend

### License
