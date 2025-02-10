import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, r2_score

# Generate synthetic Telecom Churn dataset
num_customers = 2000
telecom_data = {
    "Customer Tenure": np.random.randint(1, 72, num_customers),  # Months of subscription
    "Monthly Charges": np.random.uniform(20, 120, num_customers),  # Bill amount
    "Total Charges": np.random.uniform(100, 8000, num_customers),  # Lifetime spend
    "Customer Support Calls": np.random.randint(0, 10, num_customers),  # Number of complaints
    "Churn": np.random.choice([0, 1], size=num_customers, p=[0.75, 0.25])  # 25% churn rate
}

# Convert to DataFrame
df_churn = pd.DataFrame(telecom_data)

# Normalize numerical features
scaler = MinMaxScaler()
df_churn[["Customer Tenure", "Monthly Charges", "Total Charges", "Customer Support Calls"]] = scaler.fit_transform(
    df_churn[["Customer Tenure", "Monthly Charges", "Total Charges", "Customer Support Calls"]]
)

# Heatmap of correlations
plt.figure(figsize=(8, 6))
sns.heatmap(df_churn.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Prepare features and labels
X = df_churn.drop(columns=["Churn"])
y = df_churn["Churn"]

# Split dataset into training (1700) and testing (300)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=300/2000, random_state=42, stratify=y)

# Train Naïve Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Predict on test data
y_pred = nb_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate R2 score
r2 = r2_score(y_test, y_pred)

# Save dataset
churn_csv_path = r"C:\Users\User\Downloads\niloy\telecom_churn_dataset.csv"
df_churn.to_csv(churn_csv_path, index=False)

# Print results
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"R2 Score: {r2:.2f}")
print(f"Dataset saved as {churn_csv_path}")
