import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# 0. configure the page
st.set_page_config(
    page_title="Data Science App",
    page_icon="🧊",
    layout="wide",
)

# 1. load the data

@st.cache_data()
def load_data():
    url = 'Top_rated_movies1.csv'
    df = pd.read_csv(url, parse_dates=['release_date'])
    return df

# 2. build the UI
st.title("Data Science App")
with st.spinner("Loading data..."):
    df = load_data()

st.header("IMDB 8000 movies dataset")
c1, c2 = st.columns(2)
with c1:
    st.info("Raw Data in DataFrame")
    st.dataframe(df, use_container_width=True)

    st.success("Column information of the dataset")
    cols = df.columns.tolist()
    st.subheader(f'Total columns {len(cols)} ➡️ {", ".join(cols)}')
with c2:
    st.success("Summary of the dataset")
    st.dataframe(df.describe(), use_container_width=True)
    st.success("Data types of the dataset")
    
# 3. add some graph and widgets
st.header("Basic Data Visualization")
gop = ['bar', 'line', 'area']

c1, c2 = st.columns(2)
# column 1
sel_op = c1.selectbox("Select the type of plot for popularity", gop)
subset = df.sort_values(by='popularity')[:50]
if sel_op == gop[0]:
    fig = px.bar(subset, x='title',y='popularity', log_y=True)
elif sel_op == gop[1]:
    fig = px.line(subset, x='title',y='popularity')
elif sel_op == gop[2]:
    fig = px.area(subset, x='title',y='popularity')
c1.plotly_chart(fig, use_container_width=True)
# column 2
sel_op2 = c2.selectbox("Select the type of plot for vote count", gop)
subset2 = df.sort_values(by='vote_count')[:50]
if sel_op2 == gop[0]:
    fig = px.bar(subset2, x='title',y='vote_count', log_y=True)
elif sel_op2 == gop[1]:
    fig = px.line(subset2, x='title',y='vote_count')
elif sel_op2 == gop[2]:
    fig = px.area(subset2, x='title',y='vote_count')
c2.plotly_chart(fig, use_container_width=True)

# 4. adjust layout
t1, t2, t3 = st.tabs(["Bivariate","Trivariate", 'About'])
num_cols = df.select_dtypes(include=np.number).columns.tolist()
with t1:
    c1, c2 = st.columns(2)
    col1 = c1.radio("Select the first column for scatter plot", num_cols,)
    col2 = c2.radio("Select the Second column for scatter plot", num_cols)
    fig = px.scatter(df, x=col1, y=col2, title=f'{col1} vs {col2}')
    st.plotly_chart(fig, use_container_width=True)

with t2:
    c1, c2, c3 = st.columns(3)
    col1 = c1.selectbox("Select the first column for 3d plot", num_cols)
    col2 = c2.selectbox("Select the second column for 3d plot", num_cols)
    col3 = c3.selectbox("Select the third column for 3d plot", num_cols)
    fig = px.scatter_3d(df, x=col1, y=col2,
                        z=col3, title=f'{col1} vs {col2} vs {col3}',
                        height=700)
    st.plotly_chart(fig, use_container_width=True)

# how to run the app
# open terminal and run: 
# streamlit run main.py