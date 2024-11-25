# Import necessary libraries
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier  

st.set_page_config(page_title = "Visualizations", 
                   page_icon = "🐱")

df = pd.read_csv('pet_adoption_data.csv').drop(columns=['PetID', 'AdoptionLikelihood'])

# Title and description
st.markdown(
    """
    <h2 style = "text-align: center; color: #69503c;">Visualizations</h2>
    """,
    unsafe_allow_html = True,
)
# Create a centered layout with st.columns
col1, col2, col3 = st.columns([1, 2, 1])

with col2:  # Center column
    st.image('happy_cat.gif', use_column_width=True)

# Sidebar information
with st.sidebar:
    st.header("🐱 Visualizations")
    st.write("""
    On this page, you can:
    1. View Feature Importance plot to see which inputs have the most impact on the prediction.
    2. View Confusion Matrix to see number of accurate predictions.
    3. View Classification Report for a variety of metrics on the data.
    
    """)
    st.info("Change the model to see how the plots change!")


with st.sidebar:
    st.write('### Choose which model')
    model = st.radio('Model', options = ['Decision Tree', 'Random Forest', 'AdaBoost', 'Soft Voting'])

###PREDICTION PERFORMANCE VISUALS
if model == 'Decision Tree':
            
    # Showing additional items in tabs
    st.subheader("Model Performance and Insights")
    tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])

            # Tab 1: Feature Importance Visualization
    with tab1:
            st.write("### Feature Importance")
            st.image('feature_imp_dt.svg')
            st.caption("Features used in this prediction are ranked by relative importance.")

        # Tab 2: Confusion Matrix
    with tab2:
            st.write("### Confusion Matrix")
            st.image('adoption_dt_confusion_mat.svg')
            st.caption("Confusion Matrix of model predictions.")

        # Tab 3: Classification Report
    with tab3:
            st.write("### Classification Report")
            report_df = pd.read_csv('adoption_dt_class_report.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each species.")
       
elif model == 'Random Forest':
    # Showing additional items in tabs
    st.subheader("Model Performance and Insights")
    tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])

            # Tab 1: Feature Importance Visualization
    with tab1:
            st.write("### Feature Importance")
            st.image('feature_imp_rf.svg')
            st.caption("Features used in this prediction are ranked by relative importance.")

        # Tab 2: Confusion Matrix
    with tab2:
            st.write("### Confusion Matrix")
            st.image('adoption_rf_confusion_mat.svg')
            st.caption("Confusion Matrix of model predictions.")

        # Tab 3: Classification Report
    with tab3:
            st.write("### Classification Report")
            report_df = pd.read_csv('adoption_rf_class_report.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each species.")

elif model == 'AdaBoost':
            
    # Showing additional items in tabs
    st.subheader("Model Performance and Insights")
    tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])

            # Tab 1: Feature Importance Visualization
    with tab1:
            st.write("### Feature Importance")
            st.image('feature_imp_ab.svg')
            st.caption("Features used in this prediction are ranked by relative importance.")

        # Tab 2: Confusion Matrix
    with tab2:
            st.write("### Confusion Matrix")
            st.image('adoption_ab_confusion_mat.svg')
            st.caption("Confusion Matrix of model predictions.")

        # Tab 3: Classification Report
    with tab3:
            st.write("### Classification Report")
            report_df = pd.read_csv('adoption_ab_class_report.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each species.")

elif model == 'Soft Voting':
            
    # Showing additional items in tabs
    st.subheader("Model Performance and Insights")
    tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])

            # Tab 1: Feature Importance Visualization
    with tab1:
            st.write("### Feature Importance")
            st.image('feature_imp_svm.svg')
            st.caption("Features used in this prediction are ranked by relative importance.")

        # Tab 2: Confusion Matrix
    with tab2:
            st.write("### Confusion Matrix")
            st.image('adoption_svm_confusion_mat.svg')
            st.caption("Confusion Matrix of model predictions.")

        # Tab 3: Classification Report
    with tab3:
            st.write("### Classification Report")
            report_df = pd.read_csv('adoption_svm_class_report.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each species.")

