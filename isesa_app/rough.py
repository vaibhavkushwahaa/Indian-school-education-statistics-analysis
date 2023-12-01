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
