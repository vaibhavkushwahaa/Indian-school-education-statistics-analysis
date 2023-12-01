import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

#config
st.set_page_config(layout="wide",
                   page_title="Indian School Education Statistics Analysis",
                   page_icon='📚')


@st.cache_data()

#df_total=df[['Primary','Upper_Primary','Secondary','HrSecondary']]
def load_data():
    # Loading the dataset
    india = gpd.read_file("Indian_States.txt")
    df = pd.read_csv("dropout-ratio-2012-2015.csv")
    
    df.replace("NR", np.nan, inplace=True)

    # Converting the object datatype to float
    df["Primary"] = df["Primary"].astype(float)
    df["Upper_Primary"] = df["Upper_Primary"].astype(float)
    df["Secondary"] = df["Secondary"].astype(float)
    df["HrSecondary"] = df["HrSecondary"].astype(float)






    

    df.rename(columns={"Primary_Total": "Primary", "Upper Primary_Total": "Upper_Primary", "Secondary _Total": "Secondary", "HrSecondary_Total": "HrSecondary"}, inplace=True)
    st.sidebar.success('Dataset Loaded Successfully')
    return df
    

with st.spinner("Processing Immigration data...."):
    df=load_data()



st.title('Indian School Education Statistics Analysis')
st.sidebar.title('Indian School Education Statistics Analysis')
st.markdown('This application is a Streamlit dashboard to analyze Indian School Education Statistics')
st.sidebar.markdown('This application is a Streamlit dashboard to analyze Indian School Education Statistics')

# Plotting the correlation matrix
st.subheader("Correlation matrix of dropout rates for each level")
st.pyplot(plt.figure(figsize=(12,8)))
sns.heatmap(df_total.corr(), annot=True, cmap="Blues")
st.pyplot(plt)

# Plotting the pairplot
st.subheader("Pairplot of dropout rates for each level")
st.pyplot(sns.pairplot(df_total))
st.pyplot(plt)



