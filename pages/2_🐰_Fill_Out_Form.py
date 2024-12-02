
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

st.set_page_config(page_title = "Fill Out Form", 
                   page_icon = "🐰")


# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Title and description
st.markdown(
    """
    <h2 style = "text-align: center; color: #69503c;">Fill out the form to predict adoptability!</h2>
    """,
    unsafe_allow_html = True,
)

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


# Sidebar information
with st.sidebar:
    st.header("🐰 Form Instructions")
    st.write("""
    On this page, you can:
    1. Fill out our form to make predictions.
    2. Ensure you also select which model you would like the algorithm to use.
    2. Once filled out, your inputs will be used to predict outcome as well as prediction probability.
    """)
    st.info("📋 Tip: When all information is put in, get prediction by clicking the button **Submit Form Data**.")


# Create a sidebar for input collection
st.sidebar.header("Pet Adoption Features Input")


with st.sidebar:
    st.write('Fill Out Form', expanded = False)
    with st.form("Enter pet details manually using the form below."):
        PetType = st.selectbox('Choose animal type', options=df['PetType'].unique())
        #Breed = st.selectbox("Choose breed", options=df['Breed'].unique())
        AgeMonths = st.number_input("Enter age in months", min_value=df['AgeMonths'].min(), max_value=df['AgeMonths'].max(), step=1)
        Color = st.selectbox("Choose animals color", options=df['Color'].unique())
        Size = st.selectbox("Choose animal size", options=df['Size'].unique())
        WeightKg = st.number_input("Enter animal weight", min_value=df['WeightKg'].min(), max_value=df['WeightKg'].max(), step=0.01)
        Vaccinated = st.selectbox("Vaccinated?", options=['Yes', 'No'])
        HealthCondition = st.selectbox("Healthy animal?", options=['Healthy', 'Medical condition'])
        TimeInShelterDays = st.number_input("Enter number of days animal has spend in shelter", min_value=df['TimeInShelterDays'].min(), max_value=df['TimeInShelterDays'].max(), step=1)
        AdoptionFee = st.number_input("Enter adoption fee for animal", min_value=df['TimeInShelterDays'].min(), max_value=df['TimeInShelterDays'].max(), step=1)
        PreviousOwner = st.selectbox("Did the animal have a previous owner?", options=['Yes', 'No'])
        #model = st.radio('Model', options = ['Decision Tree', 'Random Forest', 'AdaBoost', 'Soft Voting'])
        submit_button = st.form_submit_button('Submit Form Data')

st.write('**Choose which model**')
model = st.radio('Model', options = ['Decision Tree', 'Random Forest', 'AdaBoost', 'Soft Voting'])


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
encode_df['HealthCondition'] = encode_df['HealthCondition'].replace({1: 'Medical condition', 0: 'Healthy'})
encode_df.loc[len(encode_df)] = [PetType, AgeMonths, Color, Size, WeightKg, Vaccinated, HealthCondition, TimeInShelterDays, AdoptionFee, PreviousOwner]
cat_variable = ['PetType', 'Color', 'Size', 'Vaccinated', 'HealthCondition', 'PreviousOwner']


# Create dummies for encode_df
encode_dummy_df = pd.get_dummies(encode_df, columns=cat_variable)


# Extract encoded user data
user_encoded_df = encode_dummy_df.tail(1)


st.write('Please refer to the sidebar to manually enter input features.')

if model == 'Decision Tree':
    prediction = clf_dt.predict(user_encoded_df)
    
    pred_value = prediction[0]
    prob_predictions = clf_dt.predict_proba(user_encoded_df)  # Predict probabilities
    probability_max = np.max(prob_predictions, axis=1)  # Taking the max probability

    if pred_value == 0: 
        st.metric(label = "Predicted Adoption", value = 'Not Adopted')
        st.metric(label = "Prediction Probability", value = f"{probability_max[0]*100:.2f}%")
        st.image("sad_bird.gif", caption="Better luck next time!", width = 500)
    
    elif pred_value == 1:
            st.metric(label = "Predicted Adoption", value = 'Adopted')
            st.metric(label = "Prediction Probability", value = f"{probability_max[0]*100:.2f}%")
            st.image("happy_dog.gif", caption="Congratulations!", width=500)
            st.balloons()

elif model == 'Random Forest':
    prediction = clf_rf.predict(user_encoded_df)
    pred_value = prediction[0]
    prob_predictions = clf_rf.predict_proba(user_encoded_df)  # Predict probabilities
    probability_max = np.max(prob_predictions, axis=1)  # Taking the max probability
    if pred_value == 0: 
        st.metric(label = "Predicted Adoption", value = 'Not Adopted')
        st.metric(label = "Prediction Probability", value = f"{probability_max[0]*100:.2f}%")
        st.image("sad_bird.gif", caption="Better luck next time!", width = 500)
    
    elif pred_value == 1:
            st.metric(label = "Predicted Adoption", value = 'Adopted')
            st.metric(label = "Prediction Probability", value = f"{probability_max[0]*100:.2f}%")
            st.image("happy_dog.gif", caption="Congratulations!", width=500)
            st.balloons()



elif model == 'AdaBoost':
    prediction = clf_ab.predict(user_encoded_df)
    pred_value = prediction[0]
    prob_predictions = clf_ab.predict_proba(user_encoded_df)  # Predict probabilities
    probability_max = np.max(prob_predictions, axis=1)  # Taking the max probability

    if pred_value == 0: 
        st.metric(label = "Predicted Adoption", value = 'Not Adopted')
        st.metric(label = "Prediction Probability", value = f"{probability_max[0]*100:.2f}%")
        st.image("sad_bird.gif", caption="Better luck next time!", width = 500)

    elif pred_value == 1:
            st.metric(label = "Predicted Adoption", value = 'Adopted')
            st.metric(label = "Prediction Probability", value = f"{probability_max[0]*100:.2f}%")
            st.image("happy_dog.gif", caption="Congratulations!", width=500)
            st.balloons()
    
elif model == 'Soft Voting':
    prediction = clf_svm.predict(user_encoded_df)
    pred_value = prediction[0]
    prob_predictions = clf_svm.predict_proba(user_encoded_df)  # Predict probabilities
    probability_max = np.max(prob_predictions, axis=1)  # Taking the max probability

    if pred_value == 0: 
        st.metric(label = "Predicted Adoption", value = 'Not Adopted')
        st.metric(label = "Prediction Probability", value = f"{probability_max[0]*100:.2f}%")
        st.image("sad_bird.gif", caption="Better luck next time!", width = 500)
    
    elif pred_value == 1:
            st.metric(label = "Predicted Adoption", value = 'Adopted')
            st.metric(label = "Prediction Probability", value = f"{probability_max[0]*100:.2f}%")
            st.image("happy_dog.gif", caption="Congratulations!", width=500)
            st.balloons()



