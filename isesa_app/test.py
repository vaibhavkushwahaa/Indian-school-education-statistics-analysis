import plotly.express as px
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


rename_dict = {
    "Primary_Total": "Primary",
    "Upper Primary_Total": "Upper_Primary",
    "Secondary _Total": "Secondary",
    "HrSecondary_Total": "HrSecondary",
}


@st.cache_data()
def load_enrollment_data():
    df = pd.read_csv("gross-enrollment-ratio-2013-2016.csv")
    df.replace("NR", np.nan, inplace=True)
    df.rename(columns=rename_dict, inplace=True)
    for col in ["Primary", "Upper_Primary", "Secondary", "HrSecondary"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Average"] = df[["Primary", "Upper_Primary", "Secondary", "HrSecondary"]].mean(axis=1)
    return df

with st.spinner("Processing enrollment data..."):
    df_enrollment = load_enrollment_data()

import plotly.graph_objects as go

# Assuming df_enrollment is your DataFrame and 'Year' is one of the columns
fig = go.Figure()

# Add traces for each enrollment category
fig.add_trace(go.Scatter(x=df_enrollment['Year'], y=df_enrollment['Primary'], mode='lines', name='Primary'))
fig.add_trace(go.Scatter(x=df_enrollment['Year'], y=df_enrollment['Upper_Primary'], mode='lines', name='Upper Primary'))
fig.add_trace(go.Scatter(x=df_enrollment['Year'], y=df_enrollment['Secondary'], mode='lines', name='Secondary'))
fig.add_trace(go.Scatter(x=df_enrollment['Year'], y=df_enrollment['HrSecondary'], mode='lines', name='Higher Secondary'))

# Set the title and labels
fig.update_layout(title='Gross Enrollment Ratio Over the Years',
                   xaxis_title='Year',
                   yaxis_title='Enrollment Ratio')

# Display the figure
st.plotly_chart(fig)
