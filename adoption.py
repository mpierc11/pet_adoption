
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
st.title('Pet Adoption Classification: A Machine Learning App') 

# Display an image of penguins
#st.image('animals.gif', width = 500)
st.write('Utilize our advanced Machine Learning application to predict likelihood of pet adoption.')

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
st.sidebar.header("Pet Adoption Features Input")

with st.sidebar:
    #st.image('OIP.jpg', use_column_width=True,
         #caption = "Pet Adoption Predictor")
    st.write('### Input Features')
    st.write('Either upload your data file or manually enter input features.')
    with st.expander('Choose which model'):
        model = st.radio('Model', options = ['None', 'Decision Tree', 'Random Forest', 'AdaBoost', 'Soft Voting'])
    with st.expander('Option 1: Upload CSV File', expanded = False):
        st.write('Upload a CSV file containing animal details.')
        user_csv = st.file_uploader('''Choose a CSV file''', type=['''csv'''])
        st.write('''### Sample Data Format for Upload''')
        st.dataframe(df.head(5))
        #model = st.radio('Model', options = ['Decision Tree', 'Random Forest', 'AdaBoost', 'Soft Voting'])
        st.warning('''⚠️ Ensure your uploaded file has the same column names and data types as shown above.''')


with st.sidebar:
    with st.expander('Option 2: Fill Out Form', expanded = False):
        with st.form("Enter pet details manually using the form below."):
            PetType = st.selectbox('Choose animal type', options=df['PetType'].unique())
            #Breed = st.selectbox("Choose breed", options=df['Breed'].unique())
            AgeMonths = st.number_input("Enter age in months", min_value=df['AgeMonths'].min(), max_value=df['AgeMonths'].max(), step=1)
            Color = st.selectbox("Choose animals color", options=df['Color'].unique())
            Size = st.selectbox("Choose animal size", options=df['Size'].unique())
            WeightKg = st.number_input("Enter animal weight", min_value=df['WeightKg'].min(), max_value=df['WeightKg'].max(), step=0.01)
            Vaccinated = st.selectbox("Vaccinated?", options=['Yes', 'No'])
            HealthCondition = st.selectbox("Healthy animal?", options=['Good', 'Bad'])
            TimeInShelterDays = st.number_input("Enter number of days animal has spend in shelter", min_value=df['TimeInShelterDays'].min(), max_value=df['TimeInShelterDays'].max(), step=1)
            AdoptionFee = st.number_input("Enter adoption fee for animal", min_value=df['TimeInShelterDays'].min(), max_value=df['TimeInShelterDays'].max(), step=1)
            PreviousOwner = st.selectbox("Did the animal have a previous owner?", options=['Yes', 'No'])
            #model = st.radio('Model', options = ['Decision Tree', 'Random Forest', 'AdaBoost', 'Soft Voting'])
            submit_button = st.form_submit_button('Submit Form Data')

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
encode_df.loc[len(encode_df)] = [PetType, AgeMonths, Color, Size, WeightKg, Vaccinated, HealthCondition, TimeInShelterDays, AdoptionFee, PreviousOwner]
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
        
        
if user_csv is None:
    st.write('Please refer to the sidebar to either upload your data file or manually enter input features.')
    
    if model == 'Decision Tree':
        prediction = clf_dt.predict(user_encoded_df)
        
        pred_value = prediction[0]
        if pred_value == 0: 
            st.metric(label = "Predicted Adoption", value = 'Not Adopted')
        
        elif pred_value == 1:
              st.metric(label = "Predicted Adoption", value = 'Adopted')

        prob_predictions = clf_dt.predict_proba(user_encoded_df)  # Predict probabilities
        probability_max = np.max(prob_predictions, axis=1)  # Taking the max probability

        # Add the predictions and probability columns to the DataFrame
        st.metric(label = "Prediction Probability", value = f"{probability_max[0]*100:.2f}%")

    elif model == 'Random Forest':
        prediction = clf_rf.predict(user_encoded_df)
        pred_value = prediction[0]
        if pred_value == 0: 
            st.metric(label = "Predicted Adoption", value = 'Not Adopted')
        
        elif pred_value == 1:
                st.metric(label = "Predicted Adoption", value = 'Adopted')

        prob_predictions = clf_rf.predict_proba(user_encoded_df)  # Predict probabilities
        probability_max = np.max(prob_predictions, axis=1)  # Taking the max probability

        # Add the predictions and probability columns to the DataFrame
        st.metric(label = "Prediction Probability", value = f"{probability_max[0]*100:.2f}%")


    elif model == 'AdaBoost':
        prediction = clf_ab.predict(user_encoded_df)
        
        pred_value = prediction[0]
        if pred_value == 0: 
            st.metric(label = "Predicted Adoption", value = 'Not Adopted')
        
        elif pred_value == 1:
              st.metric(label = "Predicted Adoption", value = 'Adopted')

        prob_predictions = clf_ab.predict_proba(user_encoded_df)  # Predict probabilities
        probability_max = np.max(prob_predictions, axis=1)  # Taking the max probability

        # Add the predictions and probability columns to the DataFrame
        st.metric(label = "Prediction Probability", value = f"{probability_max[0]*100:.2f}%")
        
    elif model == 'Soft Voting':
        prediction = clf_svm.predict(user_encoded_df)
        
        pred_value = prediction[0]
        if pred_value == 0: 
            st.metric(label = "Predicted Adoption", value = 'Not Adopted')
        
        elif pred_value == 1:
              st.metric(label = "Predicted Adoption", value = 'Adopted')

        prob_predictions = clf_svm.predict_proba(user_encoded_df)  # Predict probabilities
        probability_max = np.max(prob_predictions, axis=1)  # Taking the max probability

        # Add the predictions and probability columns to the DataFrame
        st.metric(label = "Prediction Probability", value = f"{probability_max[0]*100:.2f}%")



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

