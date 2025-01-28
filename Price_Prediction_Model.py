# Importing the libraries #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn import metrics

# Load the Dataset #

house = "'dataset_path'.csv" 
ham = pd.read_csv(house)

# Check for Missing Values #

print(ham.isnull().sum())

# Visualize the data using pair plots and heatmap #

sns.pairplot(ham)
plt.show() # Display the Pair Plots #

# Set the Figure Size for Heatmap #

plt.figure(figsize=(10, 8))
sns.heatmap(ham.corr(), annot=True, cmap='coolwarm')  # Create a Heatmap to Visualize Correlations #
plt.show()

# Data Preprocessing #

scaler = StandardScaler() # Create a StandardScaler object for feature scaling #
X = ham.drop("MEDV", axis=1)
y = ham["MEDV"]

# Scale the features using StandardScaler #

X_s = scaler.fit_transform(X) #

# Split the data into training and testing sets #

X_tain, X_test, y_tain, y_test = train_test_split(X_s, y, test_size = 0.17, random_state = 62)

# Create and train the linear regression model #

lireg = LinearRegression()
lireg.fit(X_tain, y_tain)

# Make predictions on the testing set #

y_pred = lireg.predict(X_test)

# Evaluate the model using Mean Squared Error and R-Squared #

ho = mean_squared_error(y_test, y_pred)
ho2 = r2_score(y_test, y_pred)
point = cross_val_score(lireg, X_s, y, cv=5, scoring = 'r2')

# Print the average R-squared score#
print(f'Crossed-Validated R-squared:{np.mean(point)}')
# Print the Mean Squared Error #
print(f'Mean Squared Error: {ho}')
# Print the R-squared score#
print(f'R-squared Score: {ho2}')

# Visualize actual vs. predicted prices #

plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual Prices vs. Predicted Prices')
plt.show()

# Print predicted and actual values #

print(y_pred)
print(y_test)

# Calculate and print additional metrics # 

print(f'MSE: {metrics.mean_squared_error(y_test, y_pred)}')
accura = lireg.score(X_test, y_test)
print(accura)
