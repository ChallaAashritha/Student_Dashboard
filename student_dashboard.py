
# Replace your entire Streamlit .py file with this updated version
# (It removes the live computation for bias-variance and loads from the precomputed pkl instead)
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, r2_score, mean_squared_error
import shap
import streamlit.components.v1 as components
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output
# List of random student names for display
random_names = [
    "Amit Sharma", "Priya Verma", "Rahul Mehta", "Neha Gupta", "Vikram Reddy",
    "Anjali Singh", "Rohan Patel", "Kiran Das", "Deepa Nair", "Sanjay Joshi"
]
# Streamlit app title
st.title("Student Performance Predictor")
st.markdown("Predict student final grades (G3) or pass/fail status using separate models.")
# Load the trained models and bias-variance data
try:
    regressor_model = joblib.load('best_rf_reg.pkl')
    classifier_model = joblib.load('best_rf_clf.pkl')
    bias_var_data = joblib.load('bias_var_data.pkl')
    st.success("Models and bias-variance data loaded successfully!")
except FileNotFoundError:
    st.error("Files not found! Ensure 'best_rf_reg.pkl', 'best_rf_clf.pkl', and 'bias_var_data.pkl' are in the same directory.")
    st.stop()
# Load dataset for feature options and test data
try:
    df = pd.read_excel('processed_student_data.xlsx', sheet_name='in')
    df['absences'] = df['absences'].astype(float)
    df['G1'] = df['G1'].astype(float)
    df['G2'] = df['G2'].astype(float)
    df['G3'] = df['G3'].astype(float)
except FileNotFoundError:
    st.error("Dataset 'processed_student_data.xlsx' not found!")
    st.stop()
# Prepare test data for visualizations
selected_features = ['traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 
                     'internet', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'sub', 'sex']
X_test = df[selected_features]
# Binary mapping as in training
binary_cols = ['schoolsup', 'famsup', 'paid', 'activities', 'internet']
for col in binary_cols:
    X_test[col] = X_test[col].map({'yes': 1, 'no': 0})
# One-hot as in training
X_test = pd.get_dummies(X_test, columns=['sub', 'sex'], drop_first=True)
y_test_reg = df['G3']
y_test_clf = (df['G3'] >= 7.2).astype(int)  # Matching training threshold
# Define expected feature order (matches training after preprocessing)
feature_order = ['traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 
                 'internet', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 
                 'sub_P', 'sex_M']  
# Categorical and numerical features (matching training)
categorical_cols_dict = {
    'schoolsup': ['yes', 'no'],
    'famsup': ['yes', 'no'],
    'paid': ['yes', 'no'],
    'activities': ['yes', 'no'],
    'internet': ['yes', 'no'],
    'sub': ['M', 'P'],
    'sex': ['F', 'M']
}
numerical_cols_dict = {
    'traveltime': (1, 4, 1),
    'studytime': (1, 4, 1),
    'failures': (0, 3, 1),
    'freetime': (1, 5, 1),
    'goout': (1, 5, 1),
    'Dalc': (1, 5, 1),
    'Walc': (1, 5, 1),
    'health': (1, 5, 1),
    'absences': (0.0, 93.0, 0.1),  # Approx max from data
    'G1': (0.0, 20.0, 0.1),
    'G2': (0.0, 20.0, 0.1)
}
# Feature names post-preprocessing
all_feature_names = feature_order
# Main tabs for Regression and Classification
reg_tab, clf_tab = st.tabs(["Regression Model (G3 Score)", "Classification Model (Pass/Fail)"])
# Regression Model Section
with reg_tab:
    st.header("Regression Model: Predict G3 Score")
    reg_single, reg_batch = st.tabs(["Single Student G3 Prediction", "Batch G3 Prediction"])
    with reg_single:
        st.subheader("Single Student G3 Prediction")
        with st.form("reg_single_form"):
            st.subheader("Categorical Features")
            input_data = {}
            cols = st.columns(3)
            for i, feature in enumerate(categorical_cols_dict):
                with cols[i % 3]:
                    input_data[feature] = st.selectbox(feature.capitalize(), categorical_cols_dict[feature], key=f"reg_cat_{feature}")
            st.subheader("Numerical Features")
            cols = st.columns(3)
            for i, feature in enumerate(numerical_cols_dict):
                with cols[i % 3]:
                    min_val, max_val, step = numerical_cols_dict[feature]
                    if step == 0.1:
                        input_data[feature] = st.number_input(feature.capitalize(), min_value=min_val, max_value=max_val, step=step, format="%.1f", key=f"reg_num_{feature}")
                    else:
                        input_data[feature] = st.slider(feature.capitalize(), min_value=int(min_val), max_value=int(max_val), step=int(step), key=f"reg_num_{feature}")
            submitted = st.form_submit_button("Predict G3")
        if submitted:
            # Preprocess input
            input_df = pd.DataFrame([input_data])
            for col in binary_cols:
                input_df[col] = input_df[col].map({'yes': 1, 'no': 0})
            input_df = pd.get_dummies(input_df, columns=['sub', 'sex'], drop_first=True)
            for col in feature_order:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[feature_order]
            try:
                predicted_score = regressor_model.predict(input_df)[0]
                st.header("Prediction Result")
                st.write(f"**Predicted G3 Score**: {predicted_score:.2f} / 20")
                # SHAP for regressor               
                try:
                    st.subheader("SHAP Explanation for G3 Score Prediction")
                    explainer_reg = shap.TreeExplainer(regressor_model)
                    shap_values_reg = explainer_reg(input_df)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    shap.plots.waterfall(shap_values_reg[0], max_display=10, show=False)
                    st.pyplot(fig, clear_figure=True)
                except Exception as e:
                    st.error(f"Error generating SHAP for regressor: {str(e)}")
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    with reg_batch:
        st.subheader("Batch G3 Prediction")
        uploaded_file = st.file_uploader("Upload student data (CSV)", type="csv", key="reg_batch_uploader")
        if uploaded_file:
            batch_df = pd.read_csv(uploaded_file)
            if set(selected_features).issubset(batch_df.columns):
                batch_df_pre = batch_df[selected_features].copy()
                for col in binary_cols:
                    batch_df_pre[col] = batch_df_pre[col].map({'yes': 1, 'no': 0})
                batch_df_pre = pd.get_dummies(batch_df_pre, columns=['sub', 'sex'], drop_first=True)
                for col in feature_order:
                    if col not in batch_df_pre.columns:
                        batch_df_pre[col] = 0
                batch_df_pre = batch_df_pre[feature_order]
                try:
                    batch_scores = regressor_model.predict(batch_df_pre)
                    # Assign random names, cycling through the list if needed
                    num_rows = len(batch_scores)
                    assigned_names = [random_names[i % len(random_names)] for i in range(num_rows)]
                    results_df = pd.DataFrame({
                        'Student Name': assigned_names,
                        'Predicted G3': np.round(batch_scores, 2)
                    })
                    st.write("Batch Prediction Results:")
                    st.dataframe(results_df)
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Results", csv, "batch_g3_predictions.csv", "text/csv")
                except Exception as e:
                    st.error(f"Error in batch prediction: {str(e)}")
            else:
                st.error(f"CSV must contain all required columns: {', '.join(selected_features)}")
# Classification Model Section
with clf_tab:
    st.header("Classification Model: Predict Pass/Fail")
    clf_single, clf_batch = st.tabs(["Single Student Pass/Fail Prediction", "Batch Pass/Fail Prediction"])
    with clf_single:
        st.subheader("Single Student Pass/Fail Prediction")
        with st.form("clf_single_form"):
            st.subheader("Categorical Features")
            input_data = {}
            cols = st.columns(3)
            for i, feature in enumerate(categorical_cols_dict):
                with cols[i % 3]:
                    input_data[feature] = st.selectbox(feature.capitalize(), categorical_cols_dict[feature], key=f"clf_cat_{feature}")
            st.subheader("Numerical Features")
            cols = st.columns(3)
            for i, feature in enumerate(numerical_cols_dict):
                with cols[i % 3]:
                    min_val, max_val, step = numerical_cols_dict[feature]
                    if step == 0.1:
                        input_data[feature] = st.number_input(feature.capitalize(), min_value=min_val, max_value=max_val, step=step, format="%.1f", key=f"clf_num_{feature}")
                    else:
                        input_data[feature] = st.slider(feature.capitalize(), min_value=int(min_val), max_value=int(max_val), step=int(step), key=f"clf_num_{feature}")
            submitted = st.form_submit_button("Predict Pass/Fail")
        if submitted:
            # Preprocess input
            input_df = pd.DataFrame([input_data])
            for col in binary_cols:
                input_df[col] = input_df[col].map({'yes': 1, 'no': 0})
            input_df = pd.get_dummies(input_df, columns=['sub', 'sex'], drop_first=True)
            for col in feature_order:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[feature_order]
            try:
                predicted_class = classifier_model.predict(input_df)[0]
                predicted_prob = classifier_model.predict_proba(input_df)[0][1]
                st.header("Prediction Result")
                st.write(f"**Predicted Pass/Fail**: {'Pass' if predicted_class == 1 else 'Fail'} (threshold >=7.2)")
                st.write(f"**Probability of Passing**: {predicted_prob*100:.2f}%")
                # SHAP for classifier           
                try:
                    st.subheader("SHAP Explanation for Pass/Fail Prediction")
                    explainer = shap.TreeExplainer(classifier_model)
                    shap_values = explainer(input_df)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    shap.plots.waterfall(shap_values[0][:, 1], max_display=10, show=False)
                    st.pyplot(fig, clear_figure=True)
                except Exception as e:
                    st.error(f"Error generating SHAP for classifier: {str(e)}")
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    with clf_batch:
        st.subheader("Batch Pass/Fail Prediction")
        uploaded_file = st.file_uploader("Upload student data (CSV)", type="csv", key="clf_batch_uploader")
        if uploaded_file:
            batch_df = pd.read_csv(uploaded_file)
            if set(selected_features).issubset(batch_df.columns):
                batch_df_pre = batch_df[selected_features].copy()
                for col in binary_cols:
                    batch_df_pre[col] = batch_df_pre[col].map({'yes': 1, 'no': 0})
                batch_df_pre = pd.get_dummies(batch_df_pre, columns=['sub', 'sex'], drop_first=True)
                for col in feature_order:
                    if col not in batch_df_pre.columns:
                        batch_df_pre[col] = 0
                batch_df_pre = batch_df_pre[feature_order]
                try:
                    batch_classes = classifier_model.predict(batch_df_pre)
                    batch_probs = classifier_model.predict_proba(batch_df_pre)[:, 1]
                    # Assign random names, cycling through the list if needed
                    num_rows = len(batch_classes)
                    assigned_names = [random_names[i % len(random_names)] for i in range(num_rows)]
                    results_df = pd.DataFrame({
                        'Student Name': assigned_names,
                        'Pass/Fail': ['Pass' if c == 1 else 'Fail' for c in batch_classes],
                        'Pass Probability (%)': np.round(batch_probs * 100, 2)
                    })
                    st.write("Batch Prediction Results:")
                    st.dataframe(results_df)
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Results", csv, "batch_passfail_predictions.csv", "text/csv")
                except Exception as e:
                    st.error(f"Error in batch prediction: {str(e)}")
            else:
                st.error(f"CSV must contain all required columns: {', '.join(selected_features)}")
# Visualizations Section
st.header("Model Visualizations")
reg_vis, clf_vis = st.tabs(["Regression Visualizations", "Classification Visualizations"])
with reg_vis:
    st.subheader("Regression Model Visualizations")
    # Preprocess X_test if not already
    X_test_pre = X_test.copy()  # Assume X_test is already preprocessed in this scope
    try:
        y_pred_reg = regressor_model.predict(X_test_pre)
    except:
        st.error("Error in generating visualizations for regression.")
    
    # Feature Importance
    try:
        importances = regressor_model.feature_importances_
        importance_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False).head(10)
        # Avoid log(0) by replacing 0 with small positive value
        importance_df['Importance'] = importance_df['Importance'].replace(0, 1e-6)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(importance_df['Feature'], importance_df['Importance'], color='#1f77b4')
        ax.set_xscale('log')  # Use log scale for better visibility
        ax.set_title('Top 10 Feature Importances (Regressor)')
        ax.set_xlabel('Log-scaled Importance')
        ax.invert_yaxis()
        st.pyplot(fig, clear_figure=True)
    except Exception as e:
        st.error(f"Error: {str(e)}")

    # Actual vs Predicted
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test_reg, y_pred_reg, alpha=0.5)
        ax.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--')
        ax.set_title(f'Actual vs Predicted G3 (R² = {r2_score(y_test_reg, y_pred_reg):.4f})')
        st.pyplot(fig, clear_figure=True)
    except Exception as e:
        st.error(f"Error: {str(e)}")
    # Residual Plot
    try:
        residuals = y_test_reg - y_pred_reg
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test_reg, residuals, alpha=0.5)
        ax.axhline(0, color='r', linestyle='--')
        ax.set_title('Residual Plot')
        st.pyplot(fig, clear_figure=True)
    except Exception as e:
        st.error(f"Error: {str(e)}")
    # SHAP Summary
    # try:
    #     explainer = shap.TreeExplainer(regressor_model)
    #     shap_values = explainer(X_test_pre)
    #     fig, ax = plt.subplots()
    #     shap.summary_plot(shap_values, X_test_pre, feature_names=all_feature_names, show=False)
    #     st.pyplot(fig, clear_figure=True)
    # except Exception as e:
    #     st.error(f"Error: {str(e)}")
    
    try:
        explainer = shap.TreeExplainer(regressor_model)
        shap_values = explainer(X_test_pre)
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test_pre, feature_names=all_feature_names, show=False)
        ax.set_xscale('symlog')  # Set x-axis to symmetric log scale
        ax.set_xlim(-10, 10)  # Set reasonable limits for symlog scale
        ax.grid(True, which="both", ls="--", alpha=0.5)  # Add gridlines for readability
        ax.minorticks_on()  # Enable minor ticks
        st.pyplot(fig, clear_figure=True)
    except Exception as e:
        st.error(f"Error: {str(e)}")
    
with clf_vis:
    st.subheader("Classification Model Visualizations")
    try:
        y_pred_clf = classifier_model.predict(X_test_pre)
        y_prob_clf = classifier_model.predict_proba(X_test_pre)[:, 1]
    except:
        st.error("Error in generating visualizations for classification.")
    
    # Feature Importance
    try:
        importances = classifier_model.feature_importances_
        importance_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False).head(10)
        # Avoid log(0) by replacing 0 with small positive value
        importance_df['Importance'] = importance_df['Importance'].replace(0, 1e-6)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(importance_df['Feature'], importance_df['Importance'], color='#ff7f0e')
        ax.set_xscale('log')  # Use log scale for better visibility
        ax.set_title('Top 10 Feature Importances (Classifier)')
        ax.set_xlabel('Log-scaled Importance')
        ax.invert_yaxis()
        st.pyplot(fig, clear_figure=True)
    except Exception as e:
        st.error(f"Error: {str(e)}")


    # Confusion Matrix
    try:
        cm = confusion_matrix(y_test_clf, y_pred_clf)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'], ax=ax)
        ax.set_title('Confusion Matrix')
        st.pyplot(fig, clear_figure=True)
    except Exception as e:
        st.error(f"Error: {str(e)}")
    # ROC Curve
    try:
        fpr, tpr, _ = roc_curve(y_test_clf, y_prob_clf)
        roc_auc = roc_auc_score(y_test_clf, y_prob_clf)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_title('ROC Curve')
        ax.legend()
        st.pyplot(fig, clear_figure=True)
    except Exception as e:
        st.error(f"Error: {str(e)}")
    # SHAP Summary
    # try:
    #     explainer = shap.TreeExplainer(classifier_model)
    #     shap_values = explainer(X_test_pre)
    #     fig, ax = plt.subplots()
    #     shap.summary_plot(shap_values[:, :, 1], X_test_pre, feature_names=all_feature_names, show=False)
    #     st.pyplot(fig, clear_figure=True)
    # except Exception as e:
    #     st.error(f"Error: {str(e)}")
    
    try:
        explainer = shap.TreeExplainer(classifier_model)
        shap_values = explainer(X_test_pre)
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values[:, :, 1], X_test_pre, feature_names=all_feature_names, show=False)
        ax.set_xscale('symlog')  # Set x-axis to symmetric log scale
        ax.set_xlim(-10, 10)  # Set reasonable limits for symlog scale
        ax.grid(True, which="both", ls="--", alpha=0.5)  # Add gridlines for readability
        ax.minorticks_on()  # Enable minor ticks
        st.pyplot(fig, clear_figure=True)
    except Exception as e:
        st.error(f"Error: {str(e)}")
# Bias-Variance Analysis Section
st.header("Bias-Variance Analysis")
reg_bias_tab, clf_bias_tab = st.tabs(["Regression", "Classification"])
with reg_bias_tab:
    st.subheader("Bias-Variance Tradeoff for Regression")
    fig, ax = plt.subplots()
    ax.plot(bias_var_data['depths'], bias_var_data['reg']['bias'], label='Bias')
    ax.plot(bias_var_data['depths'], bias_var_data['reg']['var'], label='Variance')
    ax.plot(bias_var_data['depths'], bias_var_data['reg']['error'], label='Total Error')
    ax.set_xlabel('Max Depth')
    ax.set_ylabel('Error')
    ax.set_title('Bias-Variance Tradeoff - Regression')
    ax.legend()
    st.pyplot(fig)
with clf_bias_tab:
    st.subheader("Bias-Variance Tradeoff for Classification")
    fig, ax = plt.subplots()
    ax.plot(bias_var_data['depths'], bias_var_data['clf']['bias'], label='Bias')
    ax.plot(bias_var_data['depths'], bias_var_data['clf']['var'], label='Variance')
    ax.plot(bias_var_data['depths'], bias_var_data['clf']['error'], label='Total Error (Brier Score)')
    ax.set_xlabel('Max Depth')
    ax.set_ylabel('Error')
    ax.set_title('Bias-Variance Tradeoff - Classification')
    ax.legend()
    st.pyplot(fig)
# Instructions
st.sidebar.header("Instructions")
st.sidebar.markdown("""
- Choose 'Regression Model' for G3 score predictions.
- Choose 'Classification Model' for Pass/Fail predictions.
- For single predictions, input features and predict.
- For batch, upload CSV with raw features.
- View visualizations under each model.
- Pass if G3 >= 7.2.
""")
