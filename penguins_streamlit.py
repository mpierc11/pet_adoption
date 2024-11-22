# Import necessary libraries
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Set up the title and description of the app
st.title('Penguin Classifier: A Machine Learning App') 
st.write("This app uses 6 inputs to predict the species of penguin using a model "
         "built on the Palmer's Penguin's dataset. Use the inputs in the sidebar to "
         "make your prediction!")

# Display an image of penguins
st.image('penguins.png', width = 400)

# Load the pre-trained model from the pickle file
dt_pickle = open('decision_tree_penguin.pickle', 'rb') # just want the model to 'read' the model, rb = read bytes
clf = pickle.load(dt_pickle) 
dt_pickle.close()

# Create a sidebar for input collection
st.sidebar.header('Option 1: Upload CSV Dataset')

# User input file uploader
user_csv = st.sidebar.file_uploader("Upload your CSV file here")

if user_csv is not None:
  user_df = pd.read_csv(user_csv)
  
  features = user_df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']] 
  cat_var = ['island', 'sex']
  features_encoded = pd.get_dummies(features, columns = cat_var) 

  user_prediction = clf.predict(features_encoded)

  user_df['Prediction'] = user_prediction

  st.write(user_df)

# Create a sidebar for input collection
st.sidebar.header('Option 2: Penguin Features Input')

# Sidebar input fields for categorical variables
island = st.sidebar.selectbox('Penguin Island', options = ['Biscoe', 'Dream', 'Torgerson'])
  # df = pd.read_csv('penguins.csv')
  # df.dropna(inplace = True)
  # island = st.sidebar.selectbox('Penguin Island', options = df['island'].unique())
sex = st.sidebar.selectbox('Sex', options = ['Female', 'Male'])

# Sidebar input fields for numerical variables using sliders
bill_length_mm = st.sidebar.slider('Bill Length (mm)', min_value =32.0, max_value=60.0, step=0.1)
  # min_value = df['bill_length_mm'].min(), max_value = df['bill_length_mm'].max(), step = step_diff.miin()
  # diff = df['bill_depth_mm'].sort_values().diff().dropna()
  # step_diff = diff[diff != 0]

bill_depth_mm = st.sidebar.slider('Bill Depth (mm)', min_value=13.0, max_value=21.0, step=0.1)
flipper_length_mm = st.sidebar.slider('Flipper Length (mm)', min_value=170.0, max_value=230.0, step=0.5)
body_mass_g = st.sidebar.slider('Body Mass (g)', min_value=2700.0, max_value=6300.0, step=100.0)

# Putting sex and island variables into the correct format
# so that they can be used by the model for prediction
# have to do this for all categorical variables
island_Biscoe, island_Dream, island_Torgerson = 0, 0, 0 
if island == 'Biscoe': 
  island_Biscoe = 1 
elif island == 'Dream': 
  island_Dream = 1 
elif island == 'Torgerson': 
  island_Torgerson = 1 

sex_female, sex_male = 0, 0 
if sex == 'Female': 
  sex_female = 1 
elif sex == 'Male': 
  sex_male = 1 

  #  there is another way to do this by combining user input data to training data 
  #  so you dont have to do if/elif if theres many values


# Using predict() with new data provided by the user
# Be careful: all have to be same 'type' of training data
# *Be careful: has to be in same order as training data
new_prediction = clf.predict([[bill_length_mm, bill_depth_mm, flipper_length_mm, 
  body_mass_g, island_Biscoe, island_Dream, island_Torgerson, sex_female, sex_male]]) 

# Store the predicted species
prediction_species = new_prediction[0]

# Display input summary
st.write("### Input Summary")
st.write(f"**Island**: {island}") # ** makes it bold text, * is italics
st.write(f"**Sex**: {sex}")
st.write(f"**Bill Length**: {bill_length_mm} mm")
st.write(f"**Bill Depth**: {bill_depth_mm} mm")
st.write(f"**Flipper Length**: {flipper_length_mm} mm")
st.write(f"**Body Mass**: {body_mass_g} g")

# Show the predicted species on the app
st.subheader("Predicting Your Penguin's Species")
st.success(f'We predict your penguin is of the {prediction_species} species.') # st.success displays in green bar

# Showing Feature Importance plot
# st.write('We used a machine learning model (Decision Tree) to predict the species. '
#          'The features used in this prediction are ranked by relative importance below.')
# st.image('feature_imp.svg')


# Showing additional items in tabs
st.subheader("Prediction Performance")
tab1, tab2, tab3, tab4 = st.tabs(["Decision Tree", "Feature Importance", "Confusion Matrix", "Classification Report"])

# Tab 1: Visualizing Decision Tree
with tab1:
    st.write("### Decision Tree Visualization")
    st.image('dt_visual.svg')   # make sure all visualizations from Jupyter ntbk are saved in same folder
    st.caption("Visualization of the Decision Tree used in prediction.")

# Tab 2: Feature Importance Visualization
with tab2:
    st.write("### Feature Importance")
    st.image('feature_imp.svg')
    st.caption("Features used in this prediction are ranked by relative importance.")

# Tab 3: Confusion Matrix
with tab3:
    st.write("### Confusion Matrix")
    st.image('confusion_mat.svg')
    st.caption("Confusion Matrix of model predictions.")

# Tab 4: Classification Report
with tab4:
    st.write("### Classification Report")
    report_df = pd.read_csv('class_report.csv', index_col = 0).transpose()
    st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
    st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each species.")

