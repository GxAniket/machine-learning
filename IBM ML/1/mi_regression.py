# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Optional: Force matplotlib to use TkAgg for VS Code popups
import matplotlib
matplotlib.use('TkAgg')

# Step 2: Load the CSV data
print("Reading data...")
df = pd.read_csv("bmi_data.csv")
print("Data Preview:")
print(df.head())

# Step 3: Select features and labels
x = df[['Weight (kg)']]  # independent variable (input)
y = df[['BMI']]          # dependent variable (output)

# Step 4: Train the model
model = LinearRegression()
model.fit(x, y)

# Step 5: Predict BMI values
y_pred = model.predict(x)

# Step 6: Plot original data and regression line
plt.scatter(x, y, color='blue', label='Actual BMI')
plt.plot(x, y_pred, color='red', label='Predicted BMI')
plt.xlabel("Weight (kg)")
plt.ylabel("BMI")
plt.title("Linear Regression - Weight vs BMI")
plt.legend()
plt.grid(True)
plt.show()

# Step 7: Print model evaluation
print("\nModel Evaluation:")
print("R² Score:", r2_score(y, y_pred))
print("Slope (Coefficient):", model.coef_[0][0])
print("Intercept:", model.intercept_[0])
