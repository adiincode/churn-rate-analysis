import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("customer_churn.csv")

print(df.describe())

sns.countplot(
    x="Churn",
    data=df
)

plt.title("Churn Distribution")
plt.show()
# churn by subscription 
sns.countplot(
    x="Subscription Type",
    hue="Churn",
    data=df
)

plt.show()
# payment dealy analysis 
sns.boxplot(
    x="Churn",
    y="Payment Delay",
    data=df
)

plt.show()