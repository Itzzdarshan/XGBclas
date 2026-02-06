import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# FIX FOR MAC USERS
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 1. Load Data
df = pd.read_csv('milk_quality_data.csv')

# 2. Preprocessing
le = LabelEncoder()
df['grade'] = le.fit_transform(df['grade']) # high, low, medium -> 0, 1, 2

X = df.drop('grade', axis=1)
y = df['grade']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Task 5: Optimized XGBoost Classifier
# Parameters from your assignment.ipynb
model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    colsample_bytree=0.5,
    random_state=42
)

print("Training Milk Quality XGBoost Model...")
model.fit(X_train, y_train)

# 4. Evaluation (Task 4)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"--- Final Classification Results ---")
print(f"Accuracy Score: {accuracy:.4f}")

# 5. Save Model & Metadata
model_data = {
    'model': model,
    'encoder': le,
    'features': X.columns.tolist(),
    'metrics': {'accuracy': accuracy},
    'importances': pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
}
joblib.dump(model_data, "milk_model.pkl")
print("✅ Milk Model and Metrics saved to milk_model.pkl")