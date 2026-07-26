🚗 Car Price Prediction Project

Overview

This Machine Learning project aims to predict the selling price of used cars based on various features such as manufacturing year, maximum power, transmission type, and brand. The project includes data cleaning, exploratory data analysis (EDA), model training using Linear Regression, and deployment via a Streamlit web application.

Dataset

The dataset contains historical data of over 3,500 used car sales, including both standard budget vehicles and luxury brands. The data was split rigorously, using 2,861 rows to train the model and 716 unseen rows for final testing.

Exploratory Data Analysis (EDA)

To understand the relationships between different car features, I generated a correlation matrix heatmap:

Feature Correlation Discussion

Based on the correlation heatmap, the three features with the strongest relationship to a car's selling price are its manufacturing year, maximum power, and transmission type. This aligns perfectly with real-world automotive market dynamics, as newer cars have suffered less depreciation and naturally command a higher baseline value. Furthermore, vehicles equipped with higher-power engines and automatic transmissions typically represent premium trim levels or luxury brands, which significantly drive up the final market price.

Model Performance & Insights

The model was evaluated using the 716 test vehicles and achieved the following scores:

R-squared ($R^2$): 0.7154 (The model successfully explains ~71.5% of the variance in car prices).

Mean Absolute Error (MAE): ₹143,541 (On average, the model's price prediction is off by about ₹1.43 Lakhs).

Root Mean Squared Error (RMSE): ₹302,774 (This metric heavily penalizes large errors, which naturally occur when guessing the prices of highly volatile luxury cars).

The model performed exceptionally well on standard, budget-friendly cars (e.g., Maruti, Hyundai) because the dataset provided thousands of examples to learn from.

However, the model showed more variance when predicting luxury vehicles (as reflected in the higher RMSE). This is because luxury cars have hidden premium factors (custom paint, imported leather, sports packages) that are highly unpredictable and not always captured in standard datasets.

Project Structure

Dataset/: Contains the original CSV data file.

Notebook/: Contains the main Python code for training the model.

Images/: Contains saved EDA and evaluation graphs.

app.py: The frontend and backend code for the Streamlit web app.

model.pkl & features.pkl: The saved machine learning model.

requirements.txt: Package dependencies for Streamlit deployment.
