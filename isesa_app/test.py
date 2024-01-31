# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px

# # Title for your Dashboard
# st.title('School Electricity Dashboard')

# # Load Data
# @st.cache_data
# def load_data():
#     elec = pd.read_csv('percentage-of-schools-with-electricity-2013-2016.csv')
#     return elec

# electricity = load_data()



# regions = {
#         "North India": [
#             'Haryana', 'Himachal Pradesh', 'Jammu And Kashmir', 'Punjab', 'Uttar Pradesh', 
#              'Delhi', 'Chandigarh'
#         ],
#         "South India": [
#             'Andhra Pradesh', 'Karnataka', 'Kerala', 'Tamil Nadu', 'Telangana', 
#             'Puducherry', 'Lakshadweep'
#         ],
#         "East India": [
#             'Bihar', 'Jharkhand', 'Odisha', 'West Bengal', 'Andaman & Nicobar Islands'
#         ],
#         "West India": [
#             'Goa', 'Gujarat', 'Maharashtra', 'Rajasthan', 'Dadra & Nagar Haveli', 
#             'Daman & Diu'
#         ],
#         "Central India": [
#             'Chhattisgarh', 'Madhya Pradesh', 'MADHYA PRADESH'
#         ],
#         "North East India": [
#             'Arunachal Pradesh', 'Assam', 'Manipur', 'Meghalaya', 'Mizoram', 
#             'Nagaland', 'Sikkim', 'Tripura'
#         ],
#         "Union Territories": [
#             'Pondicherry', 'Delhi', 'Chandigarh', 'Dadra & Nagar Haveli', 'Daman & Diu', 
#             'Lakshadweep', 'Andaman & Nicobar Islands', 'Puducherry'
#         ],
    
#     }

# # Create a state to region mapping
# state_to_region = {state: region for region, states in regions.items() for state in states}

# # Map each state to its region
# electricity['Region'] = electricity['State_UT'].map(state_to_region)

# # Order the regions in the DataFrame as per your defined order
# ordered_regions = list(regions.keys())
# electricity['Region'] = pd.Categorical(electricity['Region'], categories=ordered_regions, ordered=True)

# # Group by region
# region_wise_electricity = electricity.groupby('Region')['All Schools'].mean().reset_index()

# # Create the funnel chart
# fig = px.funnel(data_frame=region_wise_electricity, x='Region', y='All Schools',
#                 labels={'Region': 'Region', 'All Schools': 'Average Percentage'},
#                 color_discrete_sequence=['teal'])

# # Display the chart
# st.header("Region Wise Average Percentage of All Schools with Electricity")
# st.plotly_chart(fig, use_container_width=True)
import streamlit as st
import pandas as pd
import plotly.express as px

# Assuming 'data' is your DataFrame loaded from the CSV file
# # Load your data here instead of this line if it's not already in the environment
# data_path = 'percentage-of-schools-with-comps-2013-2016.csv'  # Update this path
# data = pd.read_csv(data_path)
# # data['year'] = pd.to_datetime(data['year'].str[:4])  # Correcting the year format

# # Streamlit app starts here
# st.title('Schools with Computers Dashboard')

# Creating a dropdown for state selection
# states = data['State_UT'].unique()
# selected_states = st.multiselect('Select State(s)', states, default=states[0])

# # Filtering data based on selected states
# filtered_data = data[data['State_UT'].isin(selected_states)]

# # Group the data by 'State_UT' and 'year' for the selected states
# grouped_data = filtered_data.groupby(['State_UT', 'year'])['All Schools'].mean().reset_index()

# # Plotting the line chart with Plotly
# fig = px.line(grouped_data, x='year', y='All Schools', color='State_UT', 
#               title='Percentage of Schools with Computers by State',
#               labels={'year': 'Year', 'All Schools': 'Percentage of Schools with Computers'})

# # Display the plot in the Streamlit app
# st.plotly_chart(fig, use_container_width=True)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Load the dataset
@st.cache_data
def load_data():
    data_path = 'percentage-of-schools-with-comps-2013-2016.csv'  # Make sure to update this path
    data = pd.read_csv(data_path)
    # data['year'] = pd.to_datetime(data['year'].str[:4])
    return data

data = load_data()

states1 = data['State_UT'].unique()
selected_states = st.multiselect('Select State(s)', states1, default=states1[0])

# Filtering data based on selected states
filtered_data = data[data['State_UT'].isin(selected_states)]

# Group the data by 'State_UT' and 'year' for the selected states
grouped_data = filtered_data.groupby(['State_UT', 'year'])['All Schools'].mean().reset_index()

# Plotting the line chart with Plotly
fig = px.line(grouped_data, x='year', y='All Schools', color='State_UT', 
              title='Percentage of Schools with Computers by State',
              labels={'year': 'Year', 'All Schools': 'Percentage of Schools with Computers'})

# Display the plot in the Streamlit app
st.plotly_chart(fig, use_container_width=True)
