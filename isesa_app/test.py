# import streamlit as st
# import pandas as pd

# # Function to load data
# def load_data():
#     data = pd.read_csv('percentage-of-schools-with-electricity-2013-2016.csv')  # Replace with your file path
#     return data

# # Analysis functions
# def get_average_primary_only(data):
#     return data['Primary_Only'].mean()

# def get_state_wise_electricity(data):
#     return data.groupby('State_UT')['All Schools'].mean().sort_values(ascending=False)

# def get_correlation_primary_sec(data):
#     return data['Primary_Only'].corr(data['Sec_Only'])

# def get_average_uprimary_only(data):
#     return data['U_Primary_Only'].mean()

# # Loading data
# data = load_data()

# # Streamlit app layout
# st.title('Schools Electricity Dashboard')

# st.header('Average Percentage of Primary Only Schools with Electricity')
# st.write(get_average_primary_only(data))

# st.header('State Wise Average Percentage of All Schools with Electricity')
# st.write(get_state_wise_electricity(data))

# st.header('Correlation between Primary Only and Secondary Only Schools')
# st.write(get_correlation_primary_sec(data))

# st.header('Average Percentage of Upper Primary Only Schools with Electricity')
# st.write(get_average_uprimary_only(data))

# # Optionally, display the data
# st.header('Data')
# st.write(data)


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Title for your Dashboard
st.title('School Electricity Dashboard')

# Load Data
@st.cache_data
def load_data():
    elec = pd.read_csv('percentage-of-schools-with-electricity-2013-2016.csv')
    return elec

electricity = load_data()

# Display Data
if st.checkbox('Show Raw Data'):
    st.subheader('Raw Data')
    st.write(electricity)



# State Wise Average Percentage of All Schools with Electricity
# state_wise_electricity = electricity.groupby('State_UT')['All Schools'].mean().sort_values(ascending=False)
# st.write("State Wise Average Percentage of All Schools with Electricity:")
# st.bar_chart(state_wise_electricity)
    
state_wise_electricity = electricity.groupby('State_UT')['All Schools'].mean()#.sort_values(ascending=False)

st.header("State Wise Average Percentage of All Schools with Electricity")
fig = px.funnel(data_frame=state_wise_electricity.reset_index(), x='State_UT', y='All Schools',
                labels={'State_UT': 'State/UT', 'All Schools': 'Average Percentage'},
                color_discrete_sequence=['teal'])

st.plotly_chart(fig, use_container_width=False)
