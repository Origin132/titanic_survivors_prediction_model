"""
Titanic Survival Prediction
============================
Use this script to predict whether a new passenger
would have survived the Titanic using the trained
Logistic Regression model.

Usage:
    python predict.py
"""

import joblib
import pandas as pd
import numpy as np


# ── Load saved model and scaler ───────────────────────────
model  = joblib.load('titanic_survival_model.pkl')
scaler = joblib.load('titanic_scaler.pkl')
print("✅ Model and scaler loaded successfully!")


# ── Define a new passenger's details ──────────────────────
# Edit these values to predict for a different passenger

new_passenger = {
    "Pclass": 1,              # 1 = First, 2 = Second, 3 = Third
    "Age": 29.0,
    "SibSp": 0,               # Number of siblings/spouses aboard
    "Parch": 0,               # Number of parents/children aboard
    "Fare": 211.34,
    "Sex_encoded": 0,         # 0 = Female, 1 = Male
    "Embarked_encoded": 0,    # 0 = Cherbourg, 1 = Queenstown, 2 = Southampton
}


# ── Convert to DataFrame ──────────────────────────────────
input_df = pd.DataFrame([new_passenger])
print("\n📋 Passenger Details:")
print(input_df.to_string(index=False))


# ── Scale and predict ─────────────────────────────────────
input_scaled = scaler.transform(input_df)
prediction   = model.predict(input_scaled)[0]
probability  = model.predict_proba(input_scaled)[0]


# ── Display result ────────────────────────────────────────
print("\n" + "=" * 40)
if prediction == 1:
    print("  🟢 Prediction: SURVIVED")
else:
    print("  🔴 Prediction: DID NOT SURVIVE")

print(f"\n  Survival Probability:     {probability[1]*100:.1f}%")
print(f"  Non-Survival Probability: {probability[0]*100:.1f}%")
print("=" * 40)
