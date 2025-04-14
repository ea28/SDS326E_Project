import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
from scipy.stats import uniform, randint
import matplotlib.pyplot as plt

# --- Configuration ---
DATA_FILE = 'SDS326E_Project/data.csv'
TARGET_VARIABLE = 'EW_r'
ID_COLUMN = 'ID'
RANDOM_SEED = 42

# Data Quality Cut Thresholds
CHISQ_PHOT_THRESHOLD = 50
SNR_THRESHOLD = 5.5
TARGET_EW_R_THRESHOLD = 500

# Final set of features determined through iteration
CORE_FEATURES = [
    'dust:Av',
    'stellar_mass',
    'sfr',
    'mass_weighted_age',
    'redshift',
    'delayed:age'
]

# Hyperparameter Search Configuration
N_ITER_SEARCH = 300
CV_FOLDS = 5

# Final Model Training Configuration
N_ESTIMATORS_FINAL_FIT = 2000
EARLY_STOPPING_ROUNDS = 50


# --- Load Data ---
df = pd.read_csv(DATA_FILE)
print(f"Loaded {DATA_FILE}. Shape: {df.shape}")

# --- Clean Target Variable ---
initial_rows = len(df)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=[TARGET_VARIABLE], inplace=True)
cleaned_rows = len(df)
if cleaned_rows < initial_rows:
    print(f"Removed {initial_rows - cleaned_rows} rows due to invalid target values ({TARGET_VARIABLE}).")

# --- Data Quality Cuts ---
print("\n--- Applying Data Quality Cuts ---")
initial_rows_before_cuts = len(df)

# Apply chisq_phot cut
df = df[df['chisq_phot'] < CHISQ_PHOT_THRESHOLD]
print(f"  Applied cut: chisq_phot < {CHISQ_PHOT_THRESHOLD}. Rows remaining: {len(df)}")

# Apply Signal-to-Noise (sn) cut
SNR_COLUMN = 'sn' # Assuming column name is 'sn'
df[SNR_COLUMN] = pd.to_numeric(df[SNR_COLUMN], errors='coerce')
rows_before_sn_nan_drop = len(df)
df.dropna(subset=[SNR_COLUMN], inplace=True) # Remove rows where SNR couldn't be converted to numeric
if len(df) < rows_before_sn_nan_drop:
    print(f"  Removed {rows_before_sn_nan_drop - len(df)} rows with NaN/invalid SNR values.")
df = df[df[SNR_COLUMN] > SNR_THRESHOLD]
print(f"  Applied cut: {SNR_COLUMN} > {SNR_THRESHOLD}. Rows remaining: {len(df)}")

# Apply EW_r cut (Target Variable)
df = df[df[TARGET_VARIABLE] < TARGET_EW_R_THRESHOLD]
print(f"  Applied cut: {TARGET_VARIABLE} < {TARGET_EW_R_THRESHOLD}. Rows remaining: {len(df)}")

final_rows_after_cuts = len(df)
rows_removed_by_cuts = initial_rows_before_cuts - final_rows_after_cuts
print(f"Total rows removed by cuts: {rows_removed_by_cuts}. Final shape: {df.shape}")

# --- Data Preparation ---
X = df[CORE_FEATURES]
y = df[TARGET_VARIABLE]
print(f"\nUsing {len(X.columns)} features for modeling: {X.columns.tolist()}")

# --- Create Train and Test Sets ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED # 80% train, 20% test
)
print(f"Data split into training ({X_train.shape[0]}) and testing ({X_test.shape[0]}) sets.")

# --- Standardize Features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Fit only on training data
X_test_scaled = scaler.transform(X_test)      # Transform test data

# Create DataFrame for SHAP plots (maintains feature names)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
print("Features standardized based on training set.")

# --- Hyperparameter Tuning (Randomized Search) ---
print("\n--- Starting Hyperparameter Tuning ---")
param_dist = {
    'n_estimators': randint(100, 1000),
    'learning_rate': uniform(0.01, 0.3),
    'max_depth': randint(3, 10),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'gamma': uniform(0, 0.5),
    'reg_alpha': uniform(0, 5),
    'reg_lambda': uniform(1, 4)
}

xgb_reg = xgb.XGBRegressor(objective='reg:squarederror',
                           random_state=RANDOM_SEED,
                           n_jobs=-1)

random_search = RandomizedSearchCV(
    estimator=xgb_reg,
    param_distributions=param_dist,
    n_iter=N_ITER_SEARCH,
    cv=CV_FOLDS,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1,
    random_state=RANDOM_SEED
)

random_search.fit(X_train_scaled, y_train)

print("Hyperparameter tuning complete.")
print(f"Best CV Score (Negative MSE): {random_search.best_score_:.4f}")
print(f"Best Parameters found: {random_search.best_params_}")

# --- Train Final Model ---
print("\n--- Training Final Model with Best Parameters & Early Stopping ---")
best_params = random_search.best_params_

# Remove n_estimators from best_params if it exists, as early stopping handles this
if 'n_estimators' in best_params:
    del best_params['n_estimators']

best_xgb_model = xgb.XGBRegressor(objective='reg:squarederror',
                                  **best_params,
                                  n_estimators=N_ESTIMATORS_FINAL_FIT,
                                  random_state=RANDOM_SEED,
                                  n_jobs=-1,
                                  early_stopping_rounds=EARLY_STOPPING_ROUNDS
                                  )

# Use test set for early stopping evaluation
fit_final_params = {
    'eval_set': [(X_test_scaled, y_test)],
    'verbose': False
}

best_xgb_model.fit(X_train_scaled, y_train, **fit_final_params)
print(f"Final model used {best_xgb_model.best_iteration} estimators (stopped based on test set performance).")

# --- Evaluate Final Model on Test Set ---
print("\n--- Evaluating Final Model on Test Set ---")
y_pred = best_xgb_model.predict(X_test_scaled)
y_test_original = y_test

# Calculate metrics
r2 = r2_score(y_test_original, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
mae = mean_absolute_error(y_test_original, y_pred)

print(f"Test Set R-squared (R2): {r2:.4f}")
print(f"Test Set Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Test Set Mean Absolute Error (MAE): {mae:.4f}")

# --- SHAP Feature Importance ---
print("\n--- Calculating SHAP Feature Importance ---")
explainer = shap.TreeExplainer(best_xgb_model)
shap_values = explainer.shap_values(X_test_scaled_df)

# Print SHAP importance table
print("\n--- Global Feature Importance (Mean Absolute SHAP Value) ---")
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_importance_df = pd.DataFrame({
    'Feature': X.columns, # Use columns from original X before scaling
    'Mean Absolute SHAP': mean_abs_shap
}).sort_values(by='Mean Absolute SHAP', ascending=False)
print(shap_importance_df.to_string())

# --- Generate and Save SHAP Summary Plots ---
print("\nGenerating SHAP summary plots...")

plt.figure()
shap.summary_plot(shap_values, X_test_scaled_df, plot_type="bar", show=False)
plt.title("Feature Importance based on SHAP Values (Best Model)")
plt.tight_layout()
plt.savefig("shap_summary_bar_best_model.png")
plt.close()
print("SHAP bar plot saved to shap_summary_bar_best_model.png")

plt.figure()
shap.summary_plot(shap_values, X_test_scaled_df, show=False)
plt.title("SHAP Summary Plot - Impact on Model Output (Best Model)")
plt.tight_layout()
plt.savefig("shap_summary_dot_best_model.png")
plt.close()
print("SHAP dot plot saved to shap_summary_dot_best_model.png")

print("\nScript finished.") 