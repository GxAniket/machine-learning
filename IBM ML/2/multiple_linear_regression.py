import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D

# Load the dataset
df = pd.read_csv("FuelConsumption.csv")

# Select relevant features
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

# Define features and target
X = cdf[['ENGINESIZE', 'FUELCONSUMPTION_COMB']]
y = cdf[['CO2EMISSIONS']]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Print evaluation metrics
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("R² Score:", r2_score(y_test, y_pred))

# Plot 3D surface
x_surf, y_surf = np.meshgrid(np.linspace(X['ENGINESIZE'].min(), X['ENGINESIZE'].max(), 100),
                             np.linspace(X['FUELCONSUMPTION_COMB'].min(), X['FUELCONSUMPTION_COMB'].max(), 100))
z_surf = model.intercept_[0] + model.coef_[0][0]*x_surf + model.coef_[0][1]*y_surf

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train['ENGINESIZE'], X_train['FUELCONSUMPTION_COMB'], y_train, color='blue', label='Train Data')
ax.plot_surface(x_surf, y_surf, z_surf, color='red', alpha=0.3)

# Labels
ax.set_xlabel('Engine Size')
ax.set_ylabel('Fuel Consumption (Combined)')
ax.set_zlabel('CO2 Emissions')
ax.set_title('Multiple Linear Regression - CO2 Emissions')
plt.legend()
plt.show()
