# Predicting Lyman-alpha Equivalent Width with XGBoost

This project predicts the rest-frame Lyman-alpha Equivalent Width ($EW_r$) of galaxies using an XGBoost model based on their physical properties. The `Model.py` script handles data filtering, hyperparameter optimization, model training, and evaluation.

## Data Preparation

The initial dataset (`data.csv`, 11862 galaxies) was filtered based on data quality:
*   Photometric fit $\chi^2_{phot} < 50$
*   Ly&alpha; Signal-to-Noise ratio $sn > 5.5$
*   Rest-frame Ly&alpha; Equivalent Width $EW_r < 500$
This resulted in a final dataset of 1965 galaxies, split into 80% training (1572) and 20% testing (393) sets. Features were standardized using `StandardScaler` fitted on the training data.

## Features

### Target Variable
*   **`EW_r`**: Rest-frame Equivalent Width (EW / (1 + $z_{Ly\alpha}$)).

### Explanatory Variables
*   **`dust:Av`**: Dust attenuation in V-band (Mag).
*   **`stellar_mass`**: Logarithm of the current stellar mass ($log_{10}(M_{\odot})$).
*   **`sfr`**: Star Formation Rate ($M_{\odot}/yr$).
*   **`mass_weighted_age`**: Mass-weighted age of the stellar population (Gyr).
*   **`redshift`**: Galaxy redshift.
*   **`delayed:age`**: Logarithm of the characteristic timescale from the delayed-$\tau$ star formation history model ($log_{10}(\tau)$ [Myr]).

## XGBoost Model Building

### Hyperparameter Tuning
Hyperparameters were tuned using `RandomizedSearchCV` because it's efficient for exploring a large parameter space compared to an exhaustive `GridSearchCV`. 
*   **Method:** 5-fold cross-validation (`CV_FOLDS=5`) was performed over 300 iterations (`N_ITER_SEARCH=300`).
*   **Optimization Metric:** The search optimized for `neg_mean_squared_error` (Negative Mean Squared Error), where a score closer to 0 is better. 
*   **Search Space:** Key parameters like `n_estimators` (100-1000), `learning_rate` (0.01-0.31), `max_depth` (3-9), `subsample` (0.6-1.0), `colsample_bytree` (0.6-1.0), `gamma` (0-0.5), `reg_alpha` (0-5), and `reg_lambda` (1-5) were explored using `randint` and `uniform` distributions (see `param_dist` in `Model.py`).
*   **Best CV Score (Negative MSE):** -4761.26
*   **Best Parameters Found:**
    ```python
    {'colsample_bytree': 0.921, 'gamma': 0.141, 'learning_rate': 0.063, 'max_depth': 3, 'n_estimators': 291, 'reg_alpha': 2.880, 'reg_lambda': 3.927, 'subsample': 0.651}
    ```
    *(Note: Float values rounded for readability. `n_estimators` found here is informational; the final model uses early stopping.)*

### Final Model Training
*   An `XGBRegressor` was initialized with the best hyperparameters found during the randomized search (excluding `n_estimators`).
*   **Training:** The model was trained on the full (scaled) training dataset.
*   **Early Stopping:** To prevent overfitting and find the optimal number of boosting rounds, early stopping (`early_stopping_rounds=50`) was used. The model's performance was evaluated on the (scaled) test set after each boosting round. Training stopped when the test set MSE did not improve for 50 consecutive rounds.
*   **Final Estimators:** The final model utilized 418 boosting rounds (estimators), determined by the early stopping process.

## Results

### Test Set Performance
*   **R²:** 0.514
*   **RMSE:** 69.55
*   **MAE:** 47.95

### Feature Importance (SHAP)
The mean absolute SHAP values for all features are:
1.  `sfr` (58.84)
2.  `dust:Av` (54.90)
3.  `mass_weighted_age` (39.28)
4.  `redshift` (30.36)
5.  `delayed:age` (18.62)
6.  `stellar_mass` (18.14)

## Visualizations

*   **SHAP Feature Importance (Bar Plot):**
    ![SHAP Feature Importance Bar Plot](shap_summary_bar_best_model.png)

*   **SHAP Summary Plot (Dot Plot):**
    ![SHAP Summary Dot Plot](shap_summary_dot_best_model.png)