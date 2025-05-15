# xgb_baseline_dmatrix.py
# ------------------------------------------------------------
# XGBoost baseline for Gross_Revenue forecasting
# uses DMatrix API so it works with any XGBoost version
# ------------------------------------------------------------
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ──────────────────────────── 1. LOAD  ────────────────────────────
df = pd.read_csv('d.csv', parse_dates=['Date']).sort_values('Date').reset_index(drop=True)

# ───────────────────── 2. FEATURE ENGINEERING ─────────────────────
# Calendar
df['month']      = df['Date'].dt.month
df['month_sin']  = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos']  = np.cos(2 * np.pi * df['month'] / 12)
df['year_norm']  = (df['Date'].dt.year - df['Date'].dt.year.min()) / (
                    df['Date'].dt.year.max() - df['Date'].dt.year.min())

# Lags
for lag in [1, 2, 3, 12]:
    df[f'lag_{lag}'] = df['Gross_Revenue'].shift(lag)

# Rolling stats (12-month window)
df['roll_mean_12'] = df['Gross_Revenue'].rolling(12).mean()
df['roll_std_12']  = df['Gross_Revenue'].rolling(12).std()

# Drop rows with NaNs introduced by lags/rolling
df = df.dropna().reset_index(drop=True)

# ───────────────────────── 3. TRAIN / TEST ────────────────────────
split_date = '2020-01-01'
train = df[df['Date'] < split_date]
test  = df[df['Date'] >= split_date]

FEATURES = [
    'Exchange_Rate_JPY_USD', 'Net_Gas_Price', 'CPI', 'Corn_Price',
    'month_sin', 'month_cos', 'year_norm',
    'lag_1', 'lag_2', 'lag_3', 'lag_12',
    'roll_mean_12', 'roll_std_12'
]

X_train, y_train = train[FEATURES], train['Gross_Revenue']
X_test,  y_test  = test[FEATURES],  test['Gross_Revenue']

# (Optional) scale numeric inputs – boosters handle raw values fine,
# but scaling helps when features differ by orders of magnitude.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest  = xgb.DMatrix(X_test,  label=y_test)

# ─────────────────────── 4. TRAIN WITH EARLY STOP ─────────────────
params = {
    'objective': 'reg:squarederror',
    'eta': 0.05,
    'max_depth': 4,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,                      # upper limit
    evals=[(dtrain, 'train'), (dtest, 'test')],
    early_stopping_rounds=20,                 # stops when test-loss hasn’t improved in 20 rounds
    verbose_eval=False
)

print(f"Best iteration: {bst.best_iteration + 1}")

# ─────────────────────────── 5. PREDICT ───────────────────────────
y_pred = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))

# ─────────────────────────── 6. METRICS ───────────────────────────
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("\nXGBoost baseline results")
print(f"  MAE : {mae:.2f}")
print(f"  RMSE: {rmse:.2f}")
