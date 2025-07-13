import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
import warnings

warnings.filterwarnings("ignore")

# Load the dataset
try:
    churn_df = pd.read_csv("ChurnData.csv")
    print("[OK] CSV loaded successfully.")
except FileNotFoundError:
    print("[ERROR] 'ChurnData.csv' not found.")
    exit()

# Normalize column names
churn_df.columns = churn_df.columns.str.strip().str.lower()

# Ensure 'churn' column exists
if 'churn' not in churn_df.columns:
    print("[ERROR] 'churn' column missing in dataset.")
    exit()

# Select relevant features
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'churn']]
churn_df['churn'] = churn_df['churn'].astype('int')

# Prepare feature matrix and labels
X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
y = np.asarray(churn_df['churn'])

# Normalize features
X_norm = StandardScaler().fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=4)

# Train logistic regression model
LR = LogisticRegression().fit(X_train, y_train)

# Predict
yhat = LR.predict(X_test)
yhat_prob = LR.predict_proba(X_test)

# Evaluate
loss = log_loss(y_test, yhat_prob)
print(f"\nLog Loss on Test Data: {loss:.4f}")

# Visualize feature coefficients
coefficients = pd.Series(LR.coef_[0], index=churn_df.columns[:-1])
coefficients.sort_values().plot(kind='barh', color='steelblue', edgecolor='black')

plt.title("Feature Coefficients in Logistic Regression Churn Model")
plt.xlabel("Coefficient Value")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
