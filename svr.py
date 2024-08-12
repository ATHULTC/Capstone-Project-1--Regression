
import streamlit as st
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def main():
    st.title("SVR Stock Price Prediction")

    # File upload
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Overview:")
        st.write(df.head())

        # Select the relevant features and target variable
        features = df.drop(columns=['Date', 'Close','Symbol', 'Series', 'Trades'])  
        target = df['Close']

        # Identify numeric and categorical columns
        numeric_cols = features.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = features.select_dtypes(include=['object']).columns

        # Create a preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())  # Standardizing numeric features
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent'))  # Impute categorical with the most frequent value
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])

        # Apply the preprocessing pipeline
        X_processed = preprocessor.fit_transform(features)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_processed, target, test_size=0.2, random_state=42)

        # Initialize the SVR model with default parameters
        svr_model = SVR(kernel='linear', C=1.0, epsilon=0.1)

        # Train the model
        svr_model.fit(X_train, y_train)

        # Make predictions on the test set using the model
        y_pred = svr_model.predict(X_test)

        # Evaluate the model
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.subheader("SVR Model Evaluation:")
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"R-squared (R²): {r2:.2f}")

        # Predict close prices for the next 5 days (if applicable)
        if len(df) >= 5:
            last_5_days_df = df.tail(5).drop(columns=['Close'])
            last_5_days_processed = preprocessor.transform(last_5_days_df)
            future_pred = svr_model.predict(last_5_days_processed)

            st.subheader("Predicted Close Prices for the Next 5 Days:")
            for i, pred in enumerate(future_pred, start=1):
                st.write(f"Day {i}: {pred:.2f}")

if __name__ == "__main__":
    main()




