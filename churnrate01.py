import pandas as pd

df = pd.read_csv("customer_churn.csv")

print(df.shape)

print(df.isnull().sum())

df.drop_duplicates(inplace=True)
print(df["Churn"].isnull().sum())
print(df[df["Churn"].isnull()])
df = df.dropna(how="all")
print(df["Churn"].isnull().sum())
