import streamlit as st

def main():
    st.set_page_config(page_title="Indian School Education Statistics Analysis", layout="wide", initial_sidebar_state="expanded")
    st.title("Indian School Education Statistics Analysis")
    
    st.sidebar.title("Pages")
    page_options = ["Dropout Ratio", "Gross Enrollment Ratio", "States with Facilities"]
    selected_page = st.sidebar.selectbox("Select a page", page_options)
    
    if selected_page == "Dropout Ratio":
        display_dropout_ratio()
    elif selected_page == "Gross Enrollment Ratio":
        display_gross_enrollment_ratio()
    elif selected_page == "States with Facilities":
        display_states_with_facilities()
    
def display_dropout_ratio():
    st.header("Dropout Ratio")
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
    # Add your code for displaying states with electricity facilities here
    
def display_toilet_facilities():
    st.subheader("States with Toilet Facilities")
    # Add your code for displaying states with toilet facilities here

if __name__ == "__main__":
    main()





#########################
# Import the libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Create a tab for the pie charts
tab1, tab2 = st.tabs(["Pie charts", "Data"])

# Write the code for the pie charts in the first tab
with tab1:
    # Create two columns for the pie charts
    c1, c2 = st.columns(2)

    # Sort the dataframe by the average dropout rate in ascending order
    df_sorted = df.sort_values(by="Average", ascending=True)

    # Select the first five rows for the best states
    df_best5 = df_sorted.head(5)

    # Select the last five rows for the worst states
    df_worst5 = df_sorted.tail(5)

    # Concatenate the best and worst states into a new dataframe
    df_comparison = pd.concat([df_best5, df_worst5])

    # Extract the data and labels for the pie chart of the best states
    data = df_best5["Average"]
    labels = df_best5["State_UT"]

    # Create a figure and an axis for the pie chart of the best states
    fig, ax = plt.subplots(figsize=(9,6))

    # Plot the pie chart of the best states
    ax.pie(data, labels=labels, autopct="%.1f%%", wedgeprops=dict(width=0.4),
            colors=['#ff9999', '#66b3ff', '#99ff99', '#ff6633', '#ffccff'],
            textprops={'fontsize': 12}, labeldistance=1.05, pctdistance=0.8)

    # Set the title for the pie chart of the best states
    ax.set_title("Pie chart of the top 5 best states in terms of dropout rates")

    # Make the pie chart of the best states equal aspect ratio
    ax.axis("equal")

    # Show the pie chart of the best states using Streamlit in the first column
    c1.pyplot(fig)

    # Extract the data and labels for the pie chart of the worst states
    data = df_worst5["Average"]
    labels = df_worst5["State_UT"]

    # Create a figure and an axis for the pie chart of the worst states
    fig, ax = plt.subplots(figsize=(9,6))

    # Plot the pie chart of the worst states
    ax.pie(data, labels=labels, autopct="%1.1f%%", 
        colors=['#ff9999', '#66b3ff', '#99ff99', '#99ffAA', '#ff6633'],
        wedgeprops=dict(width=0.4), textprops={'fontsize': 12}, labeldistance=1.05, pctdistance=0.8)

    # Set the title for the pie chart of the worst states
    ax.set_title("Pie chart of the top 5 worst states in terms of dropout rates")

    # Make the pie chart of the worst states equal aspect ratio
    ax.axis("equal")

    # Show the pie chart of the worst states using Streamlit in the second column
    c2.pyplot(fig)

# Write the code for the data in the second tab
with tab2:
    # Display the dataframe of the best and worst states
    st.dataframe(df_comparison)
