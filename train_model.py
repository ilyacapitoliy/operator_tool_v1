# train_model.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from category_encoders import CountEncoder

# 1) Load your data (already pre-cleaned, or do cleaning steps here)
df = pd.read_csv('data/offers_prepared.csv')

# 2) Prepare X, y
features = ['Brand', 'Model', 'LoadSpeed', 'Aspect', 'Rim', 'Section']
X = df[features].copy()
y = df['minPrice'].values

# 3) Handle missing values
cat_cols = ['Brand', 'Model', 'LoadSpeed']
num_cols = ['Aspect', 'Rim', 'Section']

for col in cat_cols:
    X[col] = X[col].fillna('missing')

for col in num_cols:
    median_val = X[col].median()
    X[col] = X[col].fillna(median_val)

# 4) Build a pipeline with your best params
cat_transformer = Pipeline(steps=[
    ('freq_enc', CountEncoder(cols=cat_cols))
])

num_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2))  # or remove if you don't want PCA
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])

rf_regressor = RandomForestRegressor(
    n_estimators=500,
    min_samples_leaf=1,
    max_features='sqrt',
    max_depth=None,
    random_state=42
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', rf_regressor)
])

# 5) Train/Test split (optional, just to verify performance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6) Fit
pipeline.fit(X_train, y_train)

# 7) Evaluate quickly
test_preds = pipeline.predict(X_test)
# Compute MAE, R^2, etc. - or do as you wish

# 8) Save fitted model
joblib.dump(pipeline, 'my_final_rf_model.joblib')
print("Model saved successfully!")
