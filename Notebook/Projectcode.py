import numpy as np
import matplotlib.pyplot
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,accuracy_score


df = pd.read_csv("Car Details.csv") 


# we will find and fill the null values

print(df.isnull().sum().sort_values(ascending=False))

#duplicate values dhund ke delete kar denge
print(df.duplicated().sum())
df.drop_duplicates(inplace=True)
#df.info()

#year ko calculate karenge then remove the row
df['car_age'] = 2026 - df['year']
df.drop(columns=['year'], inplace=True)
print (df)

#Extract first name of the company and drop the extra column
# Extract the first word into a new 'brand' column
df['brand'] = df['name'].str.split().str[0]

# Drop the original 'name' column
df.drop(columns=['name'], inplace=True)




# Find the 99th percentile limit
max_limit = df['km_driven'].quantile(0.99)

# Cap the values: if km_driven is greater than max_limit, make it max_limit
df['km_driven'] = np.where(df['km_driven'] > max_limit, max_limit, df['km_driven'])
print(df.head)

#Heatmap between car age, km driven and selling price

corr_matrix = df[['car_age', 'km_driven', 'selling_price']].corr()
print(corr_matrix)

# 1. Create a mask that hides the upper triangle of the grid

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# 2. Add the mask=mask parameter to your heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', mask=mask)
plt.show()

#plotting a histogram
sns.histplot(df['selling_price'], bins=50)
plt.show()

#making a scatterplot between km driven and selling price
sns.scatterplot(x='km_driven', y='selling_price', data=df)
plt.show()

#making a bar plot b/w fuel and selling price
sns.barplot(x='fuel', y='selling_price', data=df)
plt.show()

# one more bar plot between transmission and selling price
sns.barplot(x='transmission', y='selling_price', data=df)
plt.show()


df['selling_price'] = np.log(df['selling_price'])



# We include 'brand' here too since it is now a categorical feature!
df = pd.get_dummies(df, columns=['fuel', 'seller_type', 'transmission', 'owner', 'brand'], drop_first=True, dtype=int)
print(df.head())


#Defining variables

y = df['selling_price']


X = df.drop(columns=['selling_price'])


print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

from sklearn.model_selection import train_test_split

# Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("Training data rows:", X_train.shape[0])
print("Testing data rows:", X_test.shape[0])

from sklearn.linear_model import LinearRegression

# 1. Initialize the Linear Regression model
model = LinearRegression()

# 2. Train the model using your 80% training data
model.fit(X_train, y_train)

print("Model training complete!")

# 1. Ask the model to guess the prices for the test data
# Note: Because we trained it on log-prices, these guesses will be in log format
y_pred_log = model.predict(X_test)

# 2. Reverse the log transformation using np.exp (exponential)
# This turns the guesses back into real Rupees
y_pred = np.exp(y_pred_log)

# 3. Do the same for the actual answer key so we can compare apples to apples
y_test_actual = np.exp(y_test)

# 4. Let's peek at the first 5 predictions to see how well it did!
print("Predicted Prices:", np.round(y_pred[:5], 0))
print("Actual Prices:   ", np.round(y_test_actual.values[:5], 0))

# 1. Calculate the final grading metrics
# (We use the y_pred and y_test_actual variables you created in the last step)
r2 = r2_score(y_test_actual, y_pred)
mae = mean_absolute_error(y_test_actual, y_pred)
mse = mean_squared_error(y_test_actual, y_pred)
rmse = np.sqrt(mse) 

print("\n=== FINAL EXAM SCORES ===")
print(f"R-squared (R2): {r2:.4f}")
print(f"Mean Absolute Error (MAE): ₹{mae:,.2f}")
print(f"Root Mean Squared Error (RMSE): ₹{rmse:,.2f}")

# 2. Plot Predicted vs Actual Prices
plt.figure(figsize=(8, 6))
# Create the scatter plot points
sns.scatterplot(x=y_test_actual, y=y_pred, alpha=0.5, color='#2196F3')

# Draw the red "perfect prediction" diagonal line
# If a dot lands exactly on this red line, the model guessed the price perfectly.
plt.plot([y_test_actual.min(), y_test_actual.max()], 
         [y_test_actual.min(), y_test_actual.max()], 
         color='red', linestyle='--', linewidth=2)

plt.xlabel("Actual Selling Price (Rupees)")
plt.ylabel("Predicted Selling Price (Rupees)")
plt.title("Linear Regression: Actual vs. Predicted Prices")
plt.tight_layout()
plt.show()

import joblib
# Save the trained model
joblib.dump(model, 'model.pkl')

# Save the column names so our website knows what to ask the user
joblib.dump(list(X.columns), 'features.pkl')

print("Model successfully saved!")
