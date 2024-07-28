import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle

@st.cache_data
def load_example_data(dataset_name):
    return sns.load_dataset(dataset_name)

@st.cache_data
def preprocess_data(df, features, target, problem_type):
    X = df[features]
    y = df[target]
    
    # Handle missing values
    imputer = IterativeImputer()
    X = imputer.fit_transform(X)

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Encode categorical variables if classification
    if problem_type == "classification":
        if y.dtype == 'O':
            le = LabelEncoder()
            y = le.fit_transform(y)
            return X, y, le
    return X, y, None

@st.cache_data
def train_model(X_train, y_train, model_choice, problem_type):
    if problem_type == "regression":
        if model_choice == "Linear Regression":
            model = LinearRegression()
        elif model_choice == "Decision Tree":
            model = DecisionTreeRegressor()
        elif model_choice == "Random Forest":
            model = RandomForestRegressor()
        else:
            model = SVR()
    else:
        if model_choice == "Decision Tree":
            model = DecisionTreeClassifier()
        elif model_choice == "Random Forest":
            model = RandomForestClassifier()
        else:
            model = SVC()
    
    model.fit(X_train, y_train)
    return model

# Greeting and description
st.title('Machine Learning Application')
st.write('Welcome! This application helps you build and evaluate machine learning models.')

# Ask user to upload data or use example data
data_choice = st.sidebar.selectbox("Do you want to upload your own data or use an example dataset?", ("Upload Data", "Use Example Dataset"))

# Upload data section
if data_choice == "Upload Data":
    uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv", "xlsx", "tsv"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".tsv"):
            df = pd.read_csv(uploaded_file, delimiter="\t")
else:
    example_data = st.sidebar.selectbox("Choose an example dataset", ("titanic", "tips", "iris"))
    df = load_example_data(example_data)

if 'df' in locals():
    # Show basic data information
    st.write("### Data Preview")
    st.write(df.head())
    st.write("Data Shape:", df.shape)
    st.write("Data Description:", df.describe())
    st.write("Data Info:")
    st.write(df.info())
    st.write("Column Names:", df.columns.tolist())

    # Ask user to select features and target
    features = st.multiselect("Select feature columns", options=df.columns.tolist(), default=df.columns.tolist()[:-1])
    target = st.selectbox("Select target column", options=df.columns.tolist())

    # Ask user to specify problem type
    problem_type = st.radio("Specify the problem type", ("regression", "classification"))

    if features and target and problem_type:
        # Button to start analysis
        if st.button("Run Analysis"):
            X, y, le = preprocess_data(df, features, target, problem_type)

            # Train-test split
            test_size = st.slider("Select test size fraction", 0.1, 0.9, 0.2)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            # Model selection
            if problem_type == "regression":
                model_choice = st.sidebar.selectbox("Choose a regression model", ("Linear Regression", "Decision Tree", "Random Forest", "Support Vector Machine"))
            else:
                model_choice = st.sidebar.selectbox("Choose a classification model", ("Decision Tree", "Random Forest", "Support Vector Machine"))

            model = train_model(X_train, y_train, model_choice, problem_type)
            y_pred = model.predict(X_test)

            # Evaluate the model
            if problem_type == "regression":
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(y_test - y_pred))
                r2 = r2_score(y_test, y_pred)
                st.write("### Evaluation Metrics for Regression")
                st.write("Mean Squared Error:", mse)
                st.write("Root Mean Squared Error:", rmse)
                st.write("Mean Absolute Error:", mae)
                st.write("R-squared:", r2)
            else:
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                cm = confusion_matrix(y_test, y_pred)
                st.write("### Evaluation Metrics for Classification")
                st.write("Accuracy:", accuracy)
                st.write("Precision:", precision)
                st.write("Recall:", recall)
                st.write("F1 Score:", f1)
                st.write("Confusion Matrix:", cm)

            # Save the best model
            save_model = st.button("Save the Model")
            if save_model:
                with open("best_model.pkl", "wb") as f:
                    pickle.dump(model, f)
                st.write("Model saved successfully!")

            # Make predictions
            make_prediction = st.button("Make a Prediction")
            if make_prediction:
                if problem_type == "regression":
                    input_data = []
                    for feature in features:
                        value = st.number_input(f"Enter value for {feature}", value=0.0)
                        input_data.append(value)
                    input_data = scaler.transform([input_data])
                    prediction = model.predict(input_data)
                    st.write("Prediction:", prediction[0])
                else:
                    input_data = []
                    for feature in features:
                        value = st.number_input(f"Enter value for {feature}", value=0.0)
                        input_data.append(value)
                    input_data = scaler.transform([input_data])
                    prediction = model.predict(input_data)
                    st.write("Prediction:", le.inverse_transform(prediction)[0])
else:
    st.write("Please upload a dataset or select an example dataset.")

