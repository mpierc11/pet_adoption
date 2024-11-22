
# Import necessary libraries
import streamlit as st
# import pickle
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# import sklearn
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import VotingClassifier   

st.set_page_config(
    page_title = "Home",
    page_icon = "👋",
    # layout = "wide"
)

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Title and description
st.markdown(
    """
    <h2 style = "text-align: center; color: #69503c;">Pet Adoption Classification: A Machine Learning App</h2>
    """,
    unsafe_allow_html = True,
)

# Title and description
st.markdown(
    """
    <p style="text-align: center; color: #69503c; font-size: 16px;">
    Utilize our advanced Machine Learning application to predict the likelihood of pet adoption.
    </p>
    """,
    unsafe_allow_html=True,
)

st.image('animals.gif', use_column_width=True)

# Sidebar navigation header
with st.sidebar:
    st.header("🔍 Navigate the App")
    st.write("Use the links above to explore:")
    st.markdown("""
    - **Upload Data**: Upload your pet adoption CSV file.
    - **Fill Out Form**: Fill out our form to see adoption likelihood.
    - **Vizualizations**: Explore model performance and insights
    - **What Now?**: Discover next steps.
    """)

st.sidebar.info("Select a task above to proceed.")