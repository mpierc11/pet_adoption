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

st.set_page_config(page_title = "Upload Data", 
                   page_icon = "🐶")


# Title and description
st.markdown(
    """
    <h2 style = "text-align: center; color: #69503c;">Upload Your Pet Adoption Data</h2>
    <p style = "text-align: center; font-size: 18px; color: #1c2d8f;">
    Start by uploading your Adoption CSV file to unlock insights about your adoption probabilities.
    </p>
    """,
    unsafe_allow_html = True,
)

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('pet_adoption_data.csv').drop(columns=['PetID', 'AdoptionLikelihood'])
df['WeightKg'] = round(df['WeightKg'], 2)

# Load the pre-trained model from the pickle file
dt_pickle = open('decision_tree_adoption.pickle', 'rb') 
rf_pickle = open('random_forest_adoption.pickle', 'rb') 
ab_pickle = open('adaboost_adoption.pickle', 'rb') 
svm_pickle = open('svm_adoption.pickle', 'rb') 
clf_dt = pickle.load(dt_pickle) 
dt_pickle.close()
clf_rf = pickle.load(rf_pickle) 
rf_pickle.close()
clf_ab = pickle.load(ab_pickle) 
ab_pickle.close()
clf_svm = pickle.load(svm_pickle) 
svm_pickle.close()

# Create a sidebar for input collection
# Sidebar information
with st.sidebar:
    st.header("🐶 Upload Page Instructions")
    st.write("""
    On this page, you can:
    1. Upload your Pet Adoption **CSV file**
    2. Ensure the file contains valid data with columns like seen in the sample dataframe below
    3. Once uploaded, your data will be used to determine adoption likelihood.
    """)

with st.sidebar:
    st.header("Pet Adoption Features Input Example")
    st.write('''### Sample Data Format for Upload''')
    st.dataframe(df.head(5))
    st.warning('''⚠️ Ensure your uploaded file has the same column names and data types as shown above.''')

# Create a centered layout with st.columns
col1, col2, col3 = st.columns([1, 2, 1])

with col2:  # Center column
    st.image('OIP.jpg', width=500)

st.write('### Choose which model')
model = st.radio('Model', options = ['Decision Tree', 'Random Forest', 'AdaBoost', 'Soft Voting'])
st.write('Upload a CSV file containing animal details.')
user_csv = st.file_uploader('''Choose a CSV file''', type=['''csv'''])

class_mapping = {1: 'Adopted', 0: 'Not Adopted'}

# Define the color mapping for the Predicted Class column - used chat gpt to add colors to columns
def color_predicted_class(val):
    color_map = {
        'Adopted': 'background-color: #b7e4c7;',
        'Not Adopted': 'background-color: #f8d7da;',
    }
    return color_map.get(val, '') 

encode_df = df.copy()
encode_df = encode_df.drop(columns=['Breed'])
encode_df['Vaccinated'] = encode_df['Vaccinated'].replace({1: 'Yes', 0: 'No'})
encode_df['PreviousOwner'] = encode_df['PreviousOwner'].replace({1: 'Yes', 0: 'No'})
encode_df['HealthCondition'] = encode_df['HealthCondition'].replace({1: 'Good', 0: 'Bad'})
#encode_df.loc[len(encode_df)] = [PetType, AgeMonths, Color, Size, WeightKg, Vaccinated, HealthCondition, TimeInShelterDays, AdoptionFee, PreviousOwner]
cat_variable = ['PetType', 'Color', 'Size', 'Vaccinated', 'HealthCondition', 'PreviousOwner']


# Create dummies for encode_df
encode_dummy_df = pd.get_dummies(encode_df, columns=cat_variable)


# Extract encoded user data
user_encoded_df = encode_dummy_df.tail(1)

# Predictions
if user_csv is not None:
    user_df = pd.read_csv(user_csv)
    user_df = user_df.drop(columns=['Breed'])
    user_df['Vaccinated'] = user_df['Vaccinated'].replace({1: 'Yes', 0: 'No'})
    user_df['PreviousOwner'] = user_df['PreviousOwner'].replace({1: 'Yes', 0: 'No'})
    user_df['HealthCondition'] = user_df['HealthCondition'].replace({1: 'Good', 0: 'Bad'})

    encode_df = pd.concat([encode_df, user_df])
    encode_dummy_df = pd.get_dummies(encode_df)
    user_encoded_df = encode_dummy_df.tail(len(user_df))
    #user_encoded_df = user_encoded_df.drop(columns=['HealthCondition_Good', 'PreviousOwner_Yes', 'Vaccinated_Yes'])


    # Select the model based on user's choice and make predictions
    if model == 'Decision Tree':
        predictions = clf_dt.predict(user_encoded_df)  # Predict class
        prob_predictions = clf_dt.predict_proba(user_encoded_df)  # Predict probabilities
        probability_max = np.max(prob_predictions, axis=1)  # Taking the max probability

        # Add the predictions and probability columns to the DataFrame
        user_df['Predicted Class'] = [class_mapping.get(pred, pred) for pred in predictions]
        user_df['Prediction Probability %'] = [f"{x:.2f}" for x in probability_max * 100] 

        # Apply the color map to the Predicted Class column - used chat gpt for this code
        styled_df = user_df.style.applymap(color_predicted_class, subset=['Predicted Class'])
    
        # Display the updated DataFrame with predictions
        st.header('Predictions with Probabilities')
        st.dataframe(styled_df)
        
    
    elif model == 'Random Forest':
        predictions = clf_rf.predict(user_encoded_df)  # Predict class
        prob_predictions = clf_rf.predict_proba(user_encoded_df)  # Predict probabilities
        prob_predictions = prob_predictions.max(axis=1)
        
        # Add the predictions and probability columns to the DataFrame
        user_df['Predicted Class'] = [class_mapping.get(pred, pred) for pred in predictions]
        user_df['Prediction Probability %'] = [f"{x:.2f}" for x in prob_predictions * 100] # Taking the max probability

        # Apply the color map to the Predicted Class column - used chat gpt for this code
        styled_df = user_df.style.applymap(color_predicted_class, subset=['Predicted Class'])
    
        # Display the updated DataFrame with predictions
        st.header('Predictions with Probabilities')
        st.dataframe(styled_df)
        
    elif model == 'AdaBoost':
        predictions = clf_ab.predict(user_encoded_df)  # Predict class
        prob_predictions = clf_ab.predict_proba(user_encoded_df)  # Predict probabilities
        prob_predictions = prob_predictions.max(axis=1)

        # Add the predictions and probability columns to the DataFrame
        user_df['Predicted Class'] = [class_mapping.get(pred, pred) for pred in predictions]
        user_df['Prediction Probability %'] = [f"{x:.2f}" for x in prob_predictions * 100]  # Taking the max probability

        # Apply the color map to the Predicted Class column - used chat gpt for this code
        styled_df = user_df.style.applymap(color_predicted_class, subset=['Predicted Class'])
    
        # Display the updated DataFrame with predictions
        st.header('Predictions with Probabilities')
        st.dataframe(styled_df)
        

    elif model == 'Soft Voting':
        predictions = clf_svm.predict(user_encoded_df)  # Predict class
        prob_predictions = clf_svm.predict_proba(user_encoded_df)  # Predict probabilities
        prob_predictions = prob_predictions.max(axis=1)

        # Add the predictions and probability columns to the DataFrame
        user_df['Predicted Class'] = [class_mapping.get(pred, pred) for pred in predictions]
        user_df['Prediction Probability %'] = [f"{x:.2f}" for x in prob_predictions * 100]  # Taking the max probability

        # Apply the color map to the Predicted Class column - used chat gpt for this code
        styled_df = user_df.style.applymap(color_predicted_class, subset=['Predicted Class'])
    
        # Display the updated DataFrame with predictions
        st.header('Predictions with Probabilities')
        st.dataframe(styled_df)
    
    else:
        st.warning("The uploaded CSV doesn't contain any required columns for prediction. The model was run with whatever features were available.")
        