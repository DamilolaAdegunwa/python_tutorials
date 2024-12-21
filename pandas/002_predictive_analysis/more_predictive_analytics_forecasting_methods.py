import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "../../employee_dataset.xlsx"  # here is the dataset
df = pd.read_excel(file_path)
df['Join_Year'] = df['Join_Date'].dt.year


# 1. Predict Employee Performance
def predict_performance(df):
    feature_columns = ['Age', 'Salary', 'Department']
    target_column = 'Performance_Score'

    # One-hot encode categorical data
    X = df[feature_columns]
    y = df[target_column]

    categorical_features = ['Department']
    numeric_features = ['Age', 'Salary']

    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    results = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})

    return results


# 2. Forecast Hiring Trends
def forecast_hiring_trends(df):
    hiring_data = df['Join_Year'].value_counts().sort_index()
    X = np.array(hiring_data.index).reshape(-1, 1)
    y = hiring_data.values

    model = LinearRegression()
    model.fit(X, y)

    future_years = np.array(range(X.max() + 1, X.max() + 6)).reshape(-1, 1)
    forecast = model.predict(future_years)

    forecast_df = pd.DataFrame({'Year': future_years.flatten(), 'Predicted_Hires': forecast.astype(int)})

    return forecast_df


# 3. Identify High-Performing Departments
def high_performance_departments(df):
    avg_performance = df.groupby('Department')['Performance_Score'].mean()
    return avg_performance.sort_values(ascending=False)


# 4. Analyze Salary Distribution
def analyze_salary_distribution(df):
    salary_stats = df['Salary'].describe()
    return salary_stats


# 5. Predict Salary Based on Features
def predict_salary(df):
    feature_columns = ['Age', 'Department', 'Performance_Score']
    target_column = 'Salary'

    X = df[feature_columns]
    y = df[target_column]

    categorical_features = ['Department']
    numeric_features = ['Age', 'Performance_Score']

    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    results = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})

    return results


# 6. Forecast Average Performance Score per Year
def forecast_performance_trends(df):
    performance_data = df.groupby(df['Join_Year'])['Performance_Score'].mean()
    X = np.array(performance_data.index).reshape(-1, 1)
    y = performance_data.values

    model = LinearRegression()
    model.fit(X, y)

    future_years = np.array(range(X.max() + 1, X.max() + 6)).reshape(-1, 1)
    forecast = model.predict(future_years)

    forecast_df = pd.DataFrame({'Year': future_years.flatten(), 'Predicted_Performance': forecast})

    return forecast_df


# 7. Correlate performance scores with salaries to identify trends
def performance_salary_correlation(df):
    correlation = df['Performance_Score'].corr(df['Salary'])
    print(f"Correlation between Performance Score and Salary: {correlation:.2f}")

    # Scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Performance_Score', y='Salary', data=df)
    plt.title("Correlation Between Performance Score and Salary")
    plt.xlabel("Performance Score")
    plt.ylabel("Salary")
    plt.show()


# 8. Evaluate Model Performance
def evaluate_model_performance(df):
    performance_metrics = {
        'Model': ['RandomForestRegressor', 'LinearRegression'],
        'Accuracy': [0.85, 0.78]  # Placeholder values
    }
    return pd.DataFrame(performance_metrics)


# 9. Performance Clusters
def performance_clusters(df):
    from sklearn.cluster import KMeans

    X = df[['Performance_Score']]
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Performance_Cluster'] = kmeans.fit_predict(X)
    return df[['ID', 'Performance_Cluster']]


# Generate Report
def generate_report(df):
    report = {
        'Performance Predictions': predict_performance(df),
        'Hiring Trends': forecast_hiring_trends(df),
        'High-Performance Departments': high_performance_departments(df),
        'Salary Distribution': analyze_salary_distribution(df),
        'Salary Predictions': predict_salary(df),
        'Performance Trends': forecast_performance_trends(df),
        'Performance Salary Correlation': performance_salary_correlation(df),
        'Model Performance': evaluate_model_performance(df),
        'Performance Clusters': performance_clusters(df)
    }
    return report


# Main
if __name__ == "__main__":
    report = generate_report(df)
    for key, value in report.items():
        print(f"\n{key}:\n{value}\n")
