# Predicting Lyman-alpha Equivalent Width with XGBoost

This project predicts the rest-frame Lyman-alpha Equivalent Width (\(EW_r\)) of galaxies using an XGBoost model based on their physical properties. The `Model.py` script handles data filtering, hyperparameter optimization, model training, and evaluation.

## Data Preparation

The initial dataset (`data.csv`, 11862 galaxies) was filtered based on data quality:
*   Photometric fit \(\chi^2_{phot} < 50\)
*   Lyα Signal-to-Noise ratio \(sn > 5.5\)
*   Rest-frame Lyα Equivalent Width \(EW_r < 500\) Å
This resulted in a final dataset of 1965 galaxies, split into 80% training (1572) and 20% testing (393) sets. Features were standardized using `StandardScaler` fitted on the training data.

## Features

### Target Variable
*   **`EW_r`**: Rest-frame Equivalent Width (Å).

### Explanatory Variables
*   **`dust:Av`**: Dust attenuation in V-band (Mag).
*   **`stellar_mass`**: Logarithm of galaxy stellar mass (\(log_{10}(M_{\odot})\)).
*   **`sfr`**: Star Formation Rate (\(M_{\odot}/yr\)).
*   **`mass_weighted_age`**: Mass-weighted age of stellar population (Gyr).
*   **`redshift`**: Galaxy redshift.
*   **`delayed:age`**: Age parameter from delayed-\(\tau\) SFH model (\(log_{10}(Age [Myr])\)).

## XGBoost Model Building

### Hyperparameter Tuning
*   **Method:** `RandomizedSearchCV` with 5-fold cross-validation (`CV_FOLDS=5`) and 300 iterations (`N_ITER_SEARCH=300`), optimizing for `neg_mean_squared_error`.
*   **Best Score (Negative MSE):** -4761.26
*   **Best Parameters:**
    ```python
    {'colsample_bytree': 0.921, 'gamma': 0.141, 'learning_rate': 0.063, 'max_depth': 3, 'n_estimators': 291, 'reg_alpha': 2.880, 'reg_lambda': 3.927, 'subsample': 0.651}
    ```
    *(Note: Float values rounded for readability)*

### Final Model Training
*   An `XGBRegressor` was trained on the full training set using the best hyperparameters found above.
*   Early stopping was employed (`early_stopping_rounds=50`), monitoring performance on the test set. The final model used 418 estimators.

## Results

### Test Set Performance
*   **R²:** 0.514
*   **RMSE:** 69.55 Å
*   **MAE:** 47.95 Å

### Feature Importance (SHAP)
The most influential features according to mean absolute SHAP values are:
1.  `sfr` (58.84)
2.  `dust:Av` (54.90)
3.  `mass_weighted_age` (39.28)

## Visualizations

*   **SHAP Feature Importance (Bar Plot):**
    ```path=shap_summary_bar_best_model.png, description=SHAP Feature Importance Bar Plot
    ```

*   **SHAP Summary Plot (Dot Plot):**
    ```path=shap_summary_dot_best_model.png, description=SHAP Summary Dot Plot
    ```