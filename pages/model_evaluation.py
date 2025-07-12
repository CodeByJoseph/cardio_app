import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Page title
st.title("Model Evaluation for Cardiovascular Disease Prediction")

# Load and preprocess the dataset (aligned with data_browser.py and train_model.py)
@st.cache_data
def load_data():
    df = pd.read_csv('data/cardio_train.csv', sep=';')
    
    # Calculate BMI
    df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
    
    # Remove unreasonable BMIs (10 ≤ BMI ≤ 60) and outliers using IQR
    df = df[(df['bmi'] >= 10) & (df['bmi'] <= 60)]
    Q1_bmi = df['bmi'].quantile(0.25)
    Q3_bmi = df['bmi'].quantile(0.75)
    IQR_bmi = Q3_bmi - Q1_bmi
    df = df[(df['bmi'] >= Q1_bmi - 1.5 * IQR_bmi) & (df['bmi'] <= Q3_bmi + 1.5 * IQR_bmi)]
    
    # Remove unreasonable blood pressures and outliers
    df = df[(df['ap_hi'] >= 60) & (df['ap_hi'] <= 250) & 
            (df['ap_lo'] >= 40) & (df['ap_lo'] <= 150) & 
            (df['ap_hi'] > df['ap_lo'])]
    Q1_ap_hi = df['ap_hi'].quantile(0.25)
    Q3_ap_hi = df['ap_hi'].quantile(0.75)
    IQR_ap_hi = Q3_ap_hi - Q1_ap_hi
    Q1_ap_lo = df['ap_lo'].quantile(0.25)
    Q3_ap_lo = df['ap_lo'].quantile(0.75)
    IQR_ap_lo = Q3_ap_lo - Q1_ap_lo
    df = df[(df['ap_hi'] >= Q1_ap_hi - 1.5 * IQR_ap_hi) & 
            (df['ap_hi'] <= Q3_ap_hi + 1.5 * IQR_ap_hi) &
            (df['ap_lo'] >= Q1_ap_lo - 1.5 * IQR_ap_lo) & 
            (df['ap_lo'] <= Q3_ap_lo + 1.5 * IQR_ap_lo)]
    
    # Create BMI categories for Dataset 1
    def bmi_category(bmi):
        if bmi < 18.5:
            return 'Underweight'
        elif 18.5 <= bmi <= 24.9:
            return 'Normal'
        elif 25.0 <= bmi <= 29.9:
            return 'Overweight'
        elif 30.0 <= bmi <= 34.9:
            return 'Obese (Class I)'
        elif 35.0 <= bmi <= 39.9:
            return 'Obese (Class II)'
        else:
            return 'Obese (Class III)'
    df['bmi_category'] = df['bmi'].apply(bmi_category)
    
    # Create blood pressure categories for Dataset 1
    def bp_category(row):
        ap_hi, ap_lo = row['ap_hi'], row['ap_lo']
        if ap_hi >= 180 or ap_lo >= 120:
            return 'Hypertensive Crisis'
        elif ap_hi >= 140 or ap_lo >= 90:
            return 'Hypertension Stage 2'
        elif ap_hi >= 130 or ap_lo >= 80:
            return 'Hypertension Stage 1'
        elif ap_hi >= 120 and ap_lo < 80:
            return 'Elevated'
        else:
            return 'Normal'
    df['bp_category'] = df.apply(bp_category, axis=1)
    
    return df

df = load_data()

# Create two datasets
df1 = df.drop(columns=['id', 'ap_hi', 'ap_lo', 'height', 'weight', 'bmi'])
df1 = pd.get_dummies(df1, columns=['bmi_category', 'bp_category', 'gender'], drop_first=True)

df2 = df.drop(columns=['id', 'bmi_category', 'bp_category', 'height', 'weight'])
df2 = pd.get_dummies(df2, columns=['gender'], drop_first=True)

# Header for model evaluation
st.header("Model Training and Evaluation")

# Why these algorithms?
st.subheader("Why These Algorithms?")
st.markdown("""
The three algorithms evaluated—Logistic Regression, Random Forest, and XGBoost—were chosen for their complementary strengths in tackling the cardiovascular disease (CVD) prediction task:
- **Logistic Regression**: A simple, interpretable model that assumes linear relationships between features and the probability of CVD. It serves as a baseline to understand feature importance and is computationally efficient for large datasets like `cardio_train.csv`.
- **Random Forest**: An ensemble method that combines multiple decision trees to capture non-linear relationships and interactions between features (e.g., age, BMI, blood pressure). It’s robust to overfitting and handles imbalanced data well, which is relevant for CVD prediction.
- **XGBoost**: A gradient boosting algorithm that optimizes predictive performance by iteratively improving weak learners. It excels at capturing complex patterns, is highly effective for tabular data, and often outperforms other models in classification tasks like this one.
These algorithms provide a range of approaches, from simple and interpretable (Logistic Regression) to complex and powerful (XGBoost), ensuring a comprehensive evaluation of model performance.
""")

# Use full datasets for evaluation
filtered_df1 = df1
filtered_df2 = df2

# Function to train and evaluate models
@st.cache_resource
def train_and_evaluate_cached(df, dataset_name):
    X = df.drop(columns=['cardio'])
    y = df['cardio']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    if numerical_cols.size > 0:
        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    # Define algorithms and parameter grids
    models = {
        'Logistic Regression': (LogisticRegression(), {
            'C': [0.1, 1, 10],
            'solver': ['liblinear', 'lbfgs']
        }),
        'Random Forest': (RandomForestClassifier(random_state=42), {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20]
        }),
        'XGBoost': (XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), {
            'n_estimators': [50, 100],
            'max_depth': [3, 6],
            'learning_rate': [0.01, 0.1]
        })
    }
    
    results = []
    for model_name, (model, param_grid) in models.items():
        grid = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
        
        # Collect metrics
        result = {
            'Model': model_name,
            'Dataset': dataset_name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'Best Parameters': grid.best_params_
        }
        results.append(result)
    
    return pd.DataFrame(results)

# Sidebar control
st.sidebar.title("Model Evaluation Options")
run_live = st.sidebar.checkbox("💻 Run model benchmark live (slow)", value=False)

st.title("Model Evaluation for Cardiovascular Disease Prediction")

if run_live:
    with st.spinner("Training and tuning models... please wait."):
        results_df1 = train_and_evaluate_cached(df1, "Dataset 1 (Categorical)")
        results_df2 = train_and_evaluate_cached(df2, "Dataset 2 (BMI)")
        results_df = pd.concat([results_df1, results_df2], ignore_index=True)

        # Optionally save this for future use
        results_df.to_csv("data/precomputed_model_results.csv", index=False)
        st.success("Done! Results trained and loaded.")
else:
    results_df = pd.read_csv("data/precomputed_model_results.csv")
    st.info("Using precomputed results for faster performance.")

# Display results
st.subheader("Model Performance Comparison")
fig = px.bar(results_df, x='Model', y='F1 Score', color='Dataset', 
             barmode='group', 
             title="Model Performance Comparison (F1 Score) Across Datasets",
             color_discrete_sequence=['#1f77b4', '#ff7f0e'])
fig.update_layout(xaxis_title="Model", yaxis_title="F1 Score")
st.plotly_chart(fig, use_container_width=True)

# Detailed results
st.subheader("Detailed Model Evaluation Results")
st.dataframe(results_df[['Model', 'Dataset', 'Accuracy', 'Precision', 'Recall', 'F1 Score']], use_container_width=True)

# Best hyperparameters
st.subheader("Best Hyperparameters")
st.write("(GridSearchCV is used to find the best hyperparameters for each model)")
for _, row in results_df.iterrows():
    st.write(f"**{row['Model']} ({row['Dataset']}):** {row['Best Parameters']}")

# Justification
best_result = results_df.loc[results_df['F1 Score'].idxmax()]
st.markdown("""
### Model and Dataset Selection Justification
Based on the evaluation, the best-performing model is **{}** on **{}** with an F1 Score of **{:.3f}**.  
- **Dataset Choice**: Dataset 1 includes categorical features (`bmi_category`, `bp_category`, `gender`), which capture key risk factors like obesity and hypertension, potentially improving model interpretability and performance. Dataset 2 includes raw `bmi` and blood pressure values, which provide continuous data but may be less robust to outliers. The choice of **{}** is justified by its higher F1 Score, indicating better balance between precision and recall for predicting cardiovascular disease.  
- **Model Choice**: {} performed best, likely due to its ability to {} (e.g., handle non-linear relationships for Random Forest/XGBoost or simplicity for Logistic Regression). The best hyperparameters ({}) optimized the model’s performance.  
- **Implications**: The selected model and dataset suggest that {} features (e.g., categorical BMI and blood pressure) are critical for accurate CVD prediction, aligning with the multifactorial nature of CVD highlighted in the conclusion.
""".format(
    best_result['Model'], best_result['Dataset'], best_result['F1 Score'],
    best_result['Dataset'],
    best_result['Model'], 
    "capture complex patterns" if best_result['Model'] in ['Random Forest', 'XGBoost'] else "model linear relationships effectively",
    best_result['Best Parameters'],
    "categorical" if best_result['Dataset'] == "Dataset 1 (Categorical)" else "continuous"
))

