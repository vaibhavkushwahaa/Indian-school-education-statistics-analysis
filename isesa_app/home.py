import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np


#config
st.set_page_config(layout="wide",
                   page_title="Indian School Education Statistics Analysis",
                   page_icon='📚')

def load_data(path):
    df = pd.read_csv('Indian-school-education-statistics-analysis\isesa_app\gross-enrollment-ratio-2013-2016.csv') #loading the dataset to panda dataframe
    st.sidebar.success('Dataset Loaded Successfully')
    return df

with st.spinner("Processing Immigration data...."):
    df=load_data('C:\Users\pc\OneDrive\Desktop\mini project\Indian-school-education-statistics-analysis\isesa_app\gross-enrollment-ratio-2013-2016.csv')



st.title('Indian School Education Statistics Analysis')
st.sidebar.title('Indian School Education Statistics Analysis')
st.markdown('This application is a Streamlit dashboard to analyze Indian School Education Statistics')
st.sidebar.markdown('This application is a Streamlit dashboard to analyze Indian School Education Statistics')



