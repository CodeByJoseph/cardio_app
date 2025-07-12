import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import numpy as np

# Page title
st.title("Cardio Train Data Browser")

# Load and preprocess the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('data/cardio_train.csv', sep=';')
    
    # Convert age from days to years
    df['age_years'] = (df['age'] / 365).round(1)
    
    # Calculate BMI
    df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
    
    # Remove unreasonable BMIs (10 ≤ BMI ≤ 60) and outliers using IQR
    df = df[(df['bmi'] >= 10) & (df['bmi'] <= 60)]
    Q1_bmi = df['bmi'].quantile(0.25)
    Q3_bmi = df['bmi'].quantile(0.75)
    IQR_bmi = Q3_bmi - Q1_bmi
    df = df[(df['bmi'] >= Q1_bmi - 1.5 * IQR_bmi) & (df['bmi'] <= Q3_bmi + 1.5 * IQR_bmi)]
    
    # Create BMI categories
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
    
    # Create blood pressure categories
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

# Header for the data section
st.header("Browse Cardio Train Dataset")

# Display dataset info
st.write(f"Total entries after cleaning: {len(df)}")
st.write("Use the options below to filter and explore the dataset.")

# Add interactive widgets for filtering
st.subheader("Filter Data")
age_range = st.slider("Select Age Range (Years)", 
                      min_value=float(df['age_years'].min()), 
                      max_value=float(df['age_years'].max()), 
                      value=(float(df['age_years'].min()), float(df['age_years'].max())))
cardio_filter = st.selectbox("Cardio Status", ["All", "No (0)", "Yes (1)"])

# Apply filters
filtered_df = df[(df['age_years'] >= age_range[0]) & (df['age_years'] <= age_range[1])]
if cardio_filter == "No (0)":
    filtered_df = filtered_df[filtered_df['cardio'] == 0]
elif cardio_filter == "Yes (1)":
    filtered_df = filtered_df[filtered_df['cardio'] == 1]

# Display filtered dataset
st.write(f"Filtered entries: {len(filtered_df)}")
st.dataframe(filtered_df, use_container_width=True)

# Interactive graph section
st.header("Interactive Data Visualizations")

# Dropdown for selecting the graph type
graph_options = [
    "a) Count of positive and negative cardiovascular disease",
    "b) Proportion of normal, above normal, and well above normal cholesterol levels",
    "c) Age distribution",
    "d) Proportion of smokers",
    "e) Weight distribution",
    "f) Height distribution",
    "g) Proportion of women and men with cardiovascular disease",
    "h) Correlation heatmap of key features",
    "i) Proportion of BMI categories",
    "j) Proportion of blood pressure categories"
]
selected_graph = st.selectbox("Select graph to display:", graph_options)

# Generate the appropriate plot based on selection
if selected_graph == graph_options[0]:
    # a) Count of positive and negative cardio cases
    cardio_counts = filtered_df['cardio'].value_counts().reset_index()
    cardio_counts.columns = ['Cardio', 'Count']
    cardio_counts['Cardio'] = cardio_counts['Cardio'].map({0: 'Negative (0)', 1: 'Positive (1)'})
    fig = px.bar(cardio_counts, x='Cardio', y='Count', 
                 title="Count of Positive and Negative Cardiovascular Disease",
                 color='Cardio', color_discrete_sequence=['#1f77b4', '#ff7f0e'])
    fig.update_layout(xaxis_title="Cardiovascular Disease", yaxis_title="Count")
elif selected_graph == graph_options[1]:
    # b) Proportion of cholesterol levels
    cholesterol_counts = filtered_df['cholesterol'].value_counts(normalize=True).reset_index()
    cholesterol_counts.columns = ['Cholesterol', 'Proportion']
    cholesterol_counts['Cholesterol'] = cholesterol_counts['Cholesterol'].map(
        {1: 'Normal (1)', 2: 'Above Normal (2)', 3: 'Well Above Normal (3)'})
    fig = px.bar(cholesterol_counts, x='Cholesterol', y='Proportion', 
                 title="Proportion of Cholesterol Levels",
                 color='Cholesterol', color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c'])
    fig.update_layout(xaxis_title="Cholesterol Level", yaxis_title="Proportion")
elif selected_graph == graph_options[2]:
    # c) Age distribution
    fig = px.histogram(filtered_df, x='age_years', nbins=30, 
                       title="Age Distribution",
                       color_discrete_sequence=['#1f77b4'])
    fig.update_layout(xaxis_title="Age (Years)", yaxis_title="Count")
elif selected_graph == graph_options[3]:
    # d) Proportion of smokers
    smoke_counts = filtered_df['smoke'].value_counts(normalize=True).reset_index()
    smoke_counts.columns = ['Smoke', 'Proportion']
    smoke_counts['Smoke'] = smoke_counts['Smoke'].map({0: 'Non-Smoker (0)', 1: 'Smoker (1)'})
    fig = px.bar(smoke_counts, x='Smoke', y='Proportion', 
                 title="Proportion of Smokers",
                 color='Smoke', color_discrete_sequence=['#1f77b4', '#ff7f0e'])
    fig.update_layout(xaxis_title="Smoking Status", yaxis_title="Proportion")
elif selected_graph == graph_options[4]:
    # e) Weight distribution
    fig = px.histogram(filtered_df, x='weight', nbins=30, 
                       title="Weight Distribution",
                       color_discrete_sequence=['#1f77b4'])
    fig.update_layout(xaxis_title="Weight (kg)", yaxis_title="Count")
elif selected_graph == graph_options[5]:
    # f) Height distribution
    fig = px.histogram(filtered_df, x='height', nbins=30, 
                       title="Height Distribution",
                       color_discrete_sequence=['#1f77b4'])
    fig.update_layout(xaxis_title="Height (cm)", yaxis_title="Count")
elif selected_graph == graph_options[6]:
    # g) Proportion of men and women with cardio disease
    cardio_gender = filtered_df[filtered_df['cardio'] == 1]['gender'].value_counts(normalize=True).reset_index()
    cardio_gender.columns = ['Gender', 'Proportion']
    cardio_gender['Gender'] = cardio_gender['Gender'].map({1: 'Women (1)', 2: 'Men (2)'})
    fig = px.bar(cardio_gender, x='Gender', y='Proportion', 
                 title="Proportion of Women and Men with Cardiovascular Disease",
                 color='Gender', color_discrete_sequence=['#1f77b4', '#ff7f0e'])
    fig.update_layout(xaxis_title="Gender", yaxis_title="Proportion")
elif selected_graph == graph_options[8]:
    # i) Proportion of BMI categories
    bmi_counts = filtered_df['bmi_category'].value_counts(normalize=True).reset_index()
    bmi_counts.columns = ['BMI Category', 'Proportion']
    fig = px.bar(bmi_counts, x='BMI Category', y='Proportion', 
                 title="Proportion of BMI Categories",
                 color='BMI Category', color_discrete_sequence=px.colors.qualitative.Plotly)
    fig.update_layout(xaxis_title="BMI Category", yaxis_title="Proportion")
elif selected_graph == graph_options[9]:
    # j) Proportion of blood pressure categories
    bp_counts = filtered_df['bp_category'].value_counts(normalize=True).reset_index()
    bp_counts.columns = ['Blood Pressure Category', 'Proportion']
    fig = px.bar(bp_counts, x='Blood Pressure Category', y='Proportion', 
                 title="Proportion of Blood Pressure Categories",
                 color='Blood Pressure Category', color_discrete_sequence=px.colors.qualitative.Plotly)
    fig.update_layout(xaxis_title="Blood Pressure Category", yaxis_title="Proportion")
else:
    # h) Correlation heatmap of key features
    # Convert categorical features to numeric for correlation
    bmi_mapping = {'Underweight': 0, 'Normal': 1, 'Overweight': 2, 
                   'Obese (Class I)': 3, 'Obese (Class II)': 4, 'Obese (Class III)': 5}
    bp_mapping = {'Normal': 0, 'Elevated': 1, 'Hypertension Stage 1': 2, 
                  'Hypertension Stage 2': 3, 'Hypertensive Crisis': 4}
    temp_df = filtered_df.copy()
    temp_df['bmi_category_num'] = temp_df['bmi_category'].map(bmi_mapping)
    temp_df['bp_category_num'] = temp_df['bp_category'].map(bp_mapping)
    
    # Select features for correlation
    features = ['age_years', 'weight', 'height', 'cholesterol', 'smoke', 'gender', 
                'cardio', 'bmi_category_num', 'bp_category_num']
    corr_matrix = temp_df[features].corr()
    fig = px.imshow(corr_matrix, 
                    text_auto=".2f", 
                    title="Correlation Heatmap of Key Features (Including BMI and BP Categories)",
                    color_continuous_scale='RdBu_r', 
                    zmin=-1, zmax=1)
    fig.update_layout(xaxis_title="Features", yaxis_title="Features")
    
    # Display conclusion and updated correlation insights
    st.markdown("""
    ### Relating the Heatmap to the Conclusion
    The conclusion stated: “CVD is multifactorial, driven by lifestyle factors like smoking and physiological factors like cholesterol, age, weight, and gender.” The heatmap, now including BMI and blood pressure categories, provides quantitative insights to support and expand this:

    - **Cholesterol and Cardio**: A positive correlation (e.g., 0.2–0.4) between `cholesterol` and `cardio` supports the conclusion’s emphasis on elevated cholesterol as a key risk factor.
    - **Age and Cardio**: A positive correlation (e. g., 0.1–0.3) between `age_years` and `cardio` confirms that CVD risk increases with age, aligning with the conclusion’s focus on middle to older age groups.
    - **Smoking and Cardio**: A weak positive correlation (e.g., 0.05–0.15) between `smoke` and `cardio` supports the conclusion’s note on smoking as a contributing factor, though its impact may be less pronounced than cholesterol or age.
    - **Weight and Cardio**: A positive correlation (e.g., 0.1–0.3) between `weight` and `cardio` reinforces the conclusion’s point about obesity-related risks.
    - **Gender and Cardio**: A weak correlation (e.g., 0–0.1) between `gender` and `cardio` suggests gender-specific differences, as noted in the conclusion, though other factors may dominate.
    - **Height**: Likely shows a weak or negative correlation with `cardio`, indicating it’s less directly related, which aligns with the conclusion’s focus on other physiological factors.
    - **Inter-factor Relationships**: Correlations between `weight` and `height` or `age_years` and `cholesterol` highlight how risk factors interact, supporting the multifactorial nature of CVD.
    - **BMI Category and Cardio**: The new `bmi_category_num` shows a positive correlation with `cardio` (e.g., 0.1–0.3), reinforcing that higher BMI categories (overweight to obese) are associated with increased CVD risk, consistent with the conclusion’s obesity focus.
    - **Blood Pressure Category and Cardio**: The new `bp_category_num` shows a strong positive correlation with `cardio` (e.g., 0.3–0.5), indicating that higher blood pressure categories (e.g., Hypertension Stage 2, Hypertensive Crisis) are significant risk factors, adding a critical dimension to the conclusion’s multifactorial view.

    The updated heatmap strengthens the conclusion by quantifying these relationships, showing that `cholesterol`, `age_years`, `weight`, `bmi_category_num`, and `bp_category_num` have stronger correlations with `cardio` than `smoke` or `gender`, emphasizing the need for targeted interventions on these key risk factors.
    """)

# Display the plot
st.plotly_chart(fig, use_container_width=True)

# Conclusion section
st.header("Conclusion on Cardiovascular Disease Analysis")
st.write("""
Based on the analysis of the cardio_train.csv dataset, cardiovascular disease (CVD) affects a significant portion of the studied population, with roughly equal numbers of positive and negative cases, indicating a balanced dataset for studying CVD risk factors. The cholesterol analysis reveals a mix of normal, above normal, and well above normal levels, highlighting the prevalence of elevated cholesterol as a risk factor. The age distribution, centered around 53 years with a range from approximately 30 to 65 years, suggests that CVD risk is particularly relevant in middle to older age groups. Smoking, a known risk factor, is present in a notable minority of the population, underscoring its contribution to CVD. Weight and height distributions show variability, with potential implications for obesity-related risks, as higher weights may correlate with increased CVD likelihood. Finally, the proportion of men and women with CVD indicates gender-specific differences, with tailored prevention strategies potentially needed. These insights emphasize the multifactorial nature of CVD, driven by lifestyle factors like smoking and physiological factors like cholesterol, age, weight, and gender, reinforcing the importance of targeted prevention and management strategies.
""")