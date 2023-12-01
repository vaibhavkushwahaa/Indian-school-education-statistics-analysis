import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

rename_dict={"Primary_Total": "Primary", 
             "Upper Primary_Total": "Upper_Primary", 
             "Secondary _Total": "Secondary", 
             "HrSecondary_Total": "HrSecondary"}

@st.cache
def load_data():
    df = pd.read_csv('dropout-ratio-2012-2015.csv')
    india = gpd.read_file('Indian_States.txt')
    df.replace("NR", np.nan, inplace=True)
    df.rename(columns=rename_dict, inplace=True)
    df["Primary"] = df["Primary"].astype(float)
    df["Upper_Primary"] = df["Upper_Primary"].astype(float)
    df["Secondary"] = df["Secondary"].astype(float)
    df["HrSecondary"] = df["HrSecondary"].astype(float)
    df_total = df[['Primary','Upper_Primary','Secondary','HrSecondary']]
    df["Average"] = df_total.mean(axis=1)
    return df, india

def main():
    st.set_page_config(layout="wide",
                       page_title="INDIAN SCHOOL EDUCATION STATISTICS ANALYSIS APP",
                       page_icon='📚')

    st.title('Indian School Education Statistics Analysis')
    st.markdown("""This app performs analysis and visualization of the Indian School Education Statistics from 2012 to 2015.
    * **Python libraries:** pandas, streamlit, numpy, matplotlib, seaborn, geopandas
    * **Data source:** [Indian School Education Statistics](https://data.gov.in/resources/stateut-wise-average-annual-drop-out-rate-2012-13-2014-15-ministry-human-resource)""")

    st.sidebar.header('User Input')
    year = st.sidebar.selectbox('Select Year', df['year'].unique())
    state = st.sidebar.selectbox('Select State', df['State_UT'].unique())

    with st.spinner("Processing Immigration data...."):
        df, india = load_data()

    st.title('Dropout Analysis and Visualization')
    st.dataframe(df)

    # Add your code for displaying the Dropout Ratio page here

def display_gross_enrollment_ratio():
    st.header("Gross Enrollment Ratio")
    # Add your code for displaying the Gross Enrollment Ratio page here

def display_states_with_facilities():
    st.header("States with Facilities")
    facility_options = ["Computer", "Water", "Electricity", "Toilet"]
    selected_facility = st.selectbox("Select a facility", facility_options)

    if selected_facility == "Computer":
        display_computer_facilities()
    elif selected_facility == "Water":
        display_water_facilities()
    elif selected_facility == "Electricity":
        display_electricity_facilities()
    elif selected_facility == "Toilet":
        display_toilet_facilities()

def display_computer_facilities():
    st.subheader("States with Computer Facilities")
    # Add your code for displaying states with computer facilities here

def display_water_facilities():
    st.subheader("States with Water Facilities")
    # Add your code for displaying states with water facilities here

def display_electricity_facilities():
    st.subheader("States with Electricity Facilities")

def display_dropout_ratio():
    st.header("Dropout Ratio")
    # Add your code for displaying the Dropout Ratio page here



def display_computer_facilities():
    st.subheader("States with Computer Facilities")
    # Add your code for displaying states with computer facilities here








def main():
    st.sidebar.header('User Input')
    year = st.sidebar.selectbox('Select Year', df['year'].unique())
    state = st.sidebar.selectbox('Select State', df['State_UT'].unique())

    if state:
        st.header(f"Selected State: {state}")
        state_data = df[df['State_UT'] == state]
        st.write(state_data)

    if year:
        st.header(f"Selected Year: {year}")
        year_data = df[df['year'] == year]
        st.write(year_data)

    st.sidebar.title("Pages")
    page_options = ["Dropout Ratio", "Gross Enrollment Ratio", "States with Facilities"]
    selected_page = st.sidebar.selectbox("Select a page", page_options)

    if selected_page == "Dropout Ratio":
        display_dropout_ratio()
    elif selected_page == "Gross Enrollment Ratio":
        display_gross_enrollment_ratio()
    elif selected_page == "States with Facilities":
        display_states_with_facilities()

if __name__ == "__main__":
    main()