import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# LOAD DATA


df = pd.read_csv("customer_churn.csv")   # apna file name yahan likho

print("Original Shape:", df.shape)

# DATA CLEANING

# Remove rows where Churn is missing
df = df.dropna(subset=["Churn"])

# Remove any remaining NaN rows
df = df.dropna()

print("Cleaned Shape:", df.shape)
print("Total NaN:", df.isnull().sum().sum())


# ENCODING

le = LabelEncoder()

categorical_cols = [
    "Gender",
    "Subscription Type",
    "Contract Length"
]

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Ensure target is integer
df["Churn"] = df["Churn"].astype(int)

# FEATURES & TARGET


X = df.drop(["CustomerID", "Churn"], axis=1)
y = df["Churn"]


# TRAIN TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# MODEL TRAINING


model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Training Completed")

# PREDICTION


pred = model.predict(X_test)

# EVALUATION

print("\nAccuracy:")
print(accuracy_score(y_test, pred))

print("\nClassification Report:")
print(classification_report(y_test, pred))

# CONFUSION MATRIX


cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# FEATURE IMPORTANCE


importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance)

plt.figure(figsize=(10, 5))
sns.barplot(
    data=importance,
    x="Importance",
    y="Feature"
)

plt.title("Feature Importance")
plt.show()


# HIGH RISK CUSTOMERS

df["Prediction"] = model.predict(X)

high_risk = df[df["Prediction"] == 1].copy()


# RETENTION ENGINE


def retention_strategy(row):

    if row["Payment Delay"] > 15:
        return "Offer Payment Discount"

    elif row["Support Calls"] > 5:
        return "Priority Support"

    elif row["Usage Frequency"] < 3:
        return "Engagement Campaign"

    else:
        return "Loyalty Reward"

high_risk["Recommendation"] = high_risk.apply(
    retention_strategy,
    axis=1
)

print("\nHigh Risk Customers:")
print(
    high_risk[
        ["CustomerID", "Recommendation"]
    ].head()
)


# SAVE OUTPUT

high_risk.to_csv(
    "customer_retention_report.csv",
    index=False
)

print("\nRetention Report Saved Successfully")
print("File Name: customer_retention_report.csv")