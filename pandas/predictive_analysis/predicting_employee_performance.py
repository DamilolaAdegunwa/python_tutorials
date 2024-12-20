import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load the dataset
file_path = "../../expanded_employee_dataset.xlsx"
df = pd.read_excel(file_path)


# 1. Feature Engineering
def preprocess_data(df):
    df['Join_Year'] = df['Join_Date'].dt.year
    feature_columns = ['Age', 'Salary', 'Department']
    target_column = 'Performance_Score'

    # One-hot encode categorical data
    X = df[feature_columns]
    y = df[target_column]
    return X, y


# 2. Predicting Employee Performance in 9 steps
def predict_performance(df):
    # step 1: get the input and output
    X, y = preprocess_data(df)

    # step 2: Define preprocessing steps
    categorical_features = ['Department']
    numeric_features = ['Age', 'Salary']

    # step 3: for feature transformation
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    numeric_transformer = StandardScaler()

    # step 4: create the preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # step 5: Define the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # step 6: Create pipeline: It ensures that preprocessing and modeling happen in one seamless step.
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    # step 7: Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # step 8: Train the model
    pipeline.fit(X_train, y_train)
    print("Model Training Completed.")

    # step 9: Predict on the test set
    predictions = pipeline.predict(X_test)
    results = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
    print("Predictions on Test Data:")
    print(results.head())

    return pipeline


# Example Use Cases
if __name__ == "__main__":
    # Predict Employee Performance
    pipeline = predict_performance(df)