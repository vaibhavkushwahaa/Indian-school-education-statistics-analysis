import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
#config
st.set_page_config(layout="wide",
                   page_title="TEST",
                   page_icon='📚')


# Loading the dataset
india = gpd.read_file("Indian_States.txt")
#@app.route('/')  # Remove or modify this decorator
def home():
    df = pd.read_csv("dropout-ratio-2012-2015.csv")

    # Replace "NR" values with np.nan
    df.replace("NR", np.nan, inplace=True)

    # Renaming the columns
    df.rename(
        columns={
            "Primary_Total": "Primary",
            "Upper Primary_Total": "Upper_Primary",
            "Secondary _Total": "Secondary",
            "HrSecondary_Total": "HrSecondary",
        },
        inplace=True,
    )

    # Converting the object datatype to float
    df["Primary"] = df["Primary"].astype(float)
    df["Upper_Primary"] = df["Upper_Primary"].astype(float)
    df["Secondary"] = df["Secondary"].astype(float)
    df["HrSecondary"] = df["HrSecondary"].astype(float)

    # Display the dataset
    st.title("Exploring Dropout Rates in India")
    st.write("This app analyzes dropout rates in India from 2012 to 2015.")
    st.write(df.head())

    # Converting the object datatype to float
    df["Primary"] = df["Primary"].astype(float)
    df["Upper_Primary"] = df["Upper_Primary"].astype(float)
    df["Secondary"] = df["Secondary"].astype(float)
    df["HrSecondary"] = df["HrSecondary"].astype(float)

    # Display the dataset
    st.title("Exploring Dropout Rates in India")
    st.write("This app analyzes dropout rates in India from 2012 to 2015.")
    st.write(df.head())


# Plotting the correlation matrix
df_total = df[['Primary','Upper_Primary','Secondary','HrSecondary']]
plt.figure(figsize=(12,8))
sns.heatmap(df_total.corr(), annot=True, cmap="Blues",)
plt.title("Correlation matrix of dropout rates for each level")
st.pyplot(plt)


# Plotting the pairplot
sns.pairplot(df_total)
st.pyplot(plt)