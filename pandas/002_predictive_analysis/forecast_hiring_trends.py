import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = "../../expanded_employee_dataset.xlsx" # here is the dataset
df = pd.read_excel(file_path)
df['Join_Year'] = df['Join_Date'].dt.year


# 3. Forecasting Hiring Trends
def forecast_hiring_trends(df):
    # create the Join_Year column and also create the input and output (feature and target)
    hiring_data = df['Join_Year'].value_counts().sort_index()
    X = np.array(hiring_data.index).reshape(-1, 1)
    y = hiring_data.values

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Forecast next 5 years
    future_years = np.array(range(X.max() + 1, X.max() + 6)).reshape(-1, 1)
    forecast = model.predict(future_years)

    forecast_df = pd.DataFrame({'Year': future_years.flatten(), 'Predicted_Hires': forecast.astype(int)})
    print("Hiring Forecast for Next 5 Years:")
    print(forecast_df)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(X, y, label="Historical Data")
    plt.plot(future_years, forecast, label="Forecast", linestyle='--', color='orange')
    plt.xlabel("Year")
    plt.ylabel("Number of Hires")
    plt.title("Hiring Trends and Forecast")
    plt.legend()
    plt.show()

    return forecast_df


# Example Use Cases
if __name__ == "__main__":
    # Forecast Hiring Trends
    forecast_hiring_trends(df)