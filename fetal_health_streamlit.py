# https://fetalhealth-madisonpierce.streamlit.app/
# https://github.com/mpierc11/fetal_health

# Import necessary libraries
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier   

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Set up the title and description of the app
st.title('Fetal Health Classification: A Machine Learning App') 

# Display an image of penguins
st.image('fetal_health_image.gif', width = 400)
st.caption("Utilize our advanced Machine Learning Application to predict fetal health classifications.")


# Load the pre-trained model from the pickle file
dt_pickle = open('decision_tree_fetal_health.pickle', 'rb') 
rf_pickle = open('random_forest_fetal_health.pickle', 'rb') 
ab_pickle = open('adaboost_fetal_health.pickle', 'rb') 
svm_pickle = open('svm_fetal_health.pickle', 'rb') 
clf_dt = pickle.load(dt_pickle) 
dt_pickle.close()
clf_rf = pickle.load(rf_pickle) 
rf_pickle.close()
clf_ab = pickle.load(ab_pickle) 
ab_pickle.close()
clf_svm = pickle.load(svm_pickle) 
svm_pickle.close()


# Load the default dataset - for dummy encoding (properly)
default_df = pd.read_csv('fetal_health.csv')

# Create a sidebar for input collection
st.sidebar.header("Fetal Health Features Input")


with st.sidebar:
    user_csv = st.file_uploader("Upload your CSV file here")
    st.header('Sample Data Format for Upload')
    st.dataframe(default_df.head(5))
  
model = st.sidebar.radio('Model', options = ['Decision Tree', 'Random Forest', 'AdaBoost', 'Soft Voting'])

class_mapping = {1: 'Normal', 2: 'Suspect', 3: 'Pathological'}

# Define the color mapping for the Predicted Class column - used chat gpt to add colors to columns
def color_predicted_class(val):
    color_map = {
        'Normal': 'background-color: lime;',
        'Suspect': 'background-color: yellow;',
        'Pathological': 'background-color: orange;'
    }
    return color_map.get(val, '') 

# Process the uploaded CSV file
if user_csv is not None:
    user_df = pd.read_csv(user_csv)  # Load the uploaded CSV into a DataFrame

    # Reset the index of the uploaded file and default_df for consistency
    user_df.reset_index(drop=True, inplace=True)  # Reset index of uploaded CSV
   
    # Define the required feature columns that the model expects - used chat gpt to help with this syntax part
    required_columns = [
        'baseline value', 'accelerations', 'fetal_movement', 'uterine_contractions', 
        'light_decelerations', 'severe_decelerations', 'prolongued_decelerations', 
        'abnormal_short_term_variability', 'mean_value_of_short_term_variability', 
        'percentage_of_time_with_abnormal_long_term_variability', 
        'mean_value_of_long_term_variability', 'histogram_width', 'histogram_min', 
        'histogram_max', 'histogram_number_of_peaks', 'histogram_number_of_zeroes', 
        'histogram_mode', 'histogram_mean', 'histogram_median', 'histogram_variance', 
        'histogram_tendency'
    ]

    # Find the intersection of required columns and uploaded columns - used chat gpt to help with this syntax part
    available_columns = [col for col in required_columns if col in user_df.columns]

    # Check if we have any available columns for prediction
    if available_columns:
        # Select the model based on user's choice and make predictions
        if model == 'Decision Tree':
            predictions = clf_dt.predict(user_df[available_columns])  # Predict class
            prob_predictions = clf_dt.predict_proba(user_df[available_columns])  # Predict probabilities
            probability_max = np.max(prob_predictions, axis=1)  # Taking the max probability

            # Add the predictions and probability columns to the DataFrame
            user_df['Predicted Class'] = [class_mapping.get(pred, pred) for pred in predictions]
            user_df['Prediction Probability %'] = [f"{x:.2f}" for x in probability_max * 100] 

            # Apply the color map to the Predicted Class column - used chat gpt for this code
            styled_df = user_df.style.applymap(color_predicted_class, subset=['Predicted Class'])
    
           # Display the updated DataFrame with predictions
            st.header('Predictions with Probabilities')
            st.dataframe(styled_df)
            
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
                  st.image('fetal_health_dt_confusion_mat.svg')
                  st.caption("Confusion Matrix of model predictions.")

              # Tab 3: Classification Report
            with tab3:
                  st.write("### Classification Report")
                  report_df = pd.read_csv('fetal_health_dt_class_report.csv', index_col = 0).transpose()
                  st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
                  st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each species.")
       
        elif model == 'Random Forest':
            predictions = clf_rf.predict(user_df[available_columns])  # Predict class
            prob_predictions = clf_rf.predict_proba(user_df[available_columns])  # Predict probabilities
            prob_predictions = prob_predictions.max(axis=1)
            
            # Add the predictions and probability columns to the DataFrame
            user_df['Predicted Class'] = [class_mapping.get(pred, pred) for pred in predictions]
            user_df['Prediction Probability %'] = [f"{x:.2f}" for x in prob_predictions * 100] # Taking the max probability

            # Apply the color map to the Predicted Class column
            styled_df = user_df.style.applymap(color_predicted_class, subset=['Predicted Class'])

           # Display the updated DataFrame with predictions
            st.header('Predictions with Probabilities')
            st.dataframe(styled_df)
            
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
                  st.image('fetal_health_rf_confusion_mat.svg')
                  st.caption("Confusion Matrix of model predictions.")

              # Tab 3: Classification Report
            with tab3:
                  st.write("### Classification Report")
                  report_df = pd.read_csv('fetal_health_rf_class_report.csv', index_col = 0).transpose()
                  st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
                  st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each species.")
        elif model == 'AdaBoost':
            predictions = clf_ab.predict(user_df[available_columns])  # Predict class
            prob_predictions = clf_ab.predict_proba(user_df[available_columns])  # Predict probabilities
            prob_predictions = prob_predictions.max(axis=1)

            # Add the predictions and probability columns to the DataFrame
            user_df['Predicted Class'] = [class_mapping.get(pred, pred) for pred in predictions]
            user_df['Prediction Probability %'] = [f"{x:.2f}" for x in prob_predictions * 100]  # Taking the max probability

            # Apply the color map to the Predicted Class column
            styled_df = user_df.style.applymap(color_predicted_class, subset=['Predicted Class'])

        
           # Display the updated DataFrame with predictions
            st.header('Predictions with Probabilities')
            st.dataframe(styled_df)
            
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
                  st.image('fetal_health_ab_confusion_mat.svg')
                  st.caption("Confusion Matrix of model predictions.")

              # Tab 3: Classification Report
            with tab3:
                  st.write("### Classification Report")
                  report_df = pd.read_csv('fetal_health_ab_class_report.csv', index_col = 0).transpose()
                  st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
                  st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each species.")

        elif model == 'Soft Voting':
            predictions = clf_svm.predict(user_df[available_columns])  # Predict class
            prob_predictions = clf_svm.predict_proba(user_df[available_columns])  # Predict probabilities
            prob_predictions = prob_predictions.max(axis=1)

            # Add the predictions and probability columns to the DataFrame
            user_df['Predicted Class'] = [class_mapping.get(pred, pred) for pred in predictions]
            user_df['Prediction Probability %'] = [f"{x:.2f}" for x in prob_predictions * 100]  # Taking the max probability

            # Apply the color map to the Predicted Class column
            styled_df = user_df.style.applymap(color_predicted_class, subset=['Predicted Class'])

        
           # Display the updated DataFrame with predictions
            st.header('Predictions with Probabilities')
            st.dataframe(styled_df)
            
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
                  st.image('fetal_health_svm_confusion_mat.svg')
                  st.caption("Confusion Matrix of model predictions.")

              # Tab 3: Classification Report
            with tab3:
                  st.write("### Classification Report")
                  report_df = pd.read_csv('fetal_health_svm_class_report.csv', index_col = 0).transpose()
                  st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
                  st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each species.")
    else:
        st.warning("The uploaded CSV doesn't contain any required columns for prediction. The model was run with whatever features were available.")

