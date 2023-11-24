import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np


#config
st.set_page_config(layout="wide",
                   page_title="Indian School Education Statistics Analysis",
                   page_icon='📚')


@st.cache_data()


def load_data():
    df = pd.read_csv('gross-enrollment-ratio-2013-2016.csv') #loading the dataset to panda dataframe
    st.sidebar.success('Dataset Loaded Successfully')
    return df

with st.spinner("Processing Immigration data...."):
    df=load_data()



st.title('Indian School Education Statistics Analysis')
st.sidebar.title('Indian School Education Statistics Analysis')
st.markdown('This application is a Streamlit dashboard to analyze Indian School Education Statistics')
st.sidebar.markdown('This application is a Streamlit dashboard to analyze Indian School Education Statistics')



