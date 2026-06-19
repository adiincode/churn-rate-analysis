# import pandas as pd

# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score

# df = pd.read_csv(
#     "customer_churn.csv"
# )

# X = df.drop(
#     ["CustomerID", "Churn"],
#     axis=1
# )

# y = df["Churn"]

# X_train, X_test, y_train, y_test = (
#     train_test_split(
#         X,
#         y,
#         test_size=0.2,
#         random_state=42
#     )
# )

# model = RandomForestClassifier(
#     n_estimators=200,
#     random_state=42
# )

# model.fit(X_train, y_train)

# pred = model.predict(X_test)

# print(
#     "Accuracy:",
#     accuracy_score(
#         y_test,
#         pred
#     )
# )
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("customer_churn.csv")

print("Original Shape:", df.shape)

# Remove rows where Churn is missing
df = df.dropna(subset=["Churn"])

# Remove any remaining NaN rows
df = df.dropna()

print("Shape After Cleaning:", df.shape)

# Verify no missing values
print("NaN in Churn:", df["Churn"].isnull().sum())
print("Total NaN:", df.isnull().sum().sum())

# Ensure target is integer
df["Churn"] = df["Churn"].astype(int)

# Features and Target
X = df.drop(["CustomerID", "Churn"], axis=1)
y = df["Churn"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)

# Results
print("\nAccuracy:")
print(accuracy_score(y_test, pred))

print("\nClassification Report:")
print(classification_report(y_test, pred))