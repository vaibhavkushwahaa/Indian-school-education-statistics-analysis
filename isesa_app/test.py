import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
@st.cache_data()
def load_data():
    df = pd.read_csv('gross-enrollment-ratio-2013-2016.csv')
    df['Year'] = df['Year'].apply(lambda x: int(x.split('-')[0]))  # Convert Year to integer

    

    return df

df = load_data()


st.title("Indian School Education Enrollment Analysis (2013-2016)")



