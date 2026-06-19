import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("customer_churn.csv")

le = LabelEncoder()

cat_cols = [
    "Gender",
    "Subscription Type",
    "Contract Length"
]

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

df.to_csv(
    "customer_churn.csv",
    index=False
)

print("Encoding Complete")
print("Shape:", df.shape)

print("NaN in Churn:")
print(df["Churn"].isnull().sum())

print(df[df["Churn"].isnull()])
