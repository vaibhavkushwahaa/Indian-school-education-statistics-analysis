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

c1,c2=st.columns(2)
percentage_with_electricity = electricity['All Schools'].mean()

# Calculate the percentage of schools without electricity
percentage_without_electricity = 100 - percentage_with_electricity

# Data for pie chart
pie_data = {
    'Category': ['With Electricity', 'Without Electricity'],
    'Percentage': [percentage_with_electricity, percentage_without_electricity]
}

df_pie = pd.DataFrame(pie_data)

# Create the pie chart
fig = px.pie(df_pie, names='Category', values='Percentage', title='Electricity Availability in Schools')

# Display the pie chart
c1.plotly_chart(fig)

#################

yearly_electricity = electricity.groupby('year')['All Schools'].mean().reset_index()

# Create the line chart
fig = px.line(yearly_electricity, x='year', y='All Schools', 
              title='Rate of Electricity Availability in Schools (2013-2016)',
              labels={'year': 'Year', 'All Schools': 'Percentage with Electricity'})

# Display the line chart
c2.plotly_chart(fig)
################
# line_chart_data = {
#     'Year': ['2013-14','2014-15','2015-16'],
#     'Percentage with Electricity': [
#         electricity['2013-14'].mean(), 
#         electricity['2014-15'].mean(), 
#         electricity['2015-16'].mean()
#     ]
# }

# df_line = pd.DataFrame(line_chart_data)

# # Create the line chart
# fig = px.line(df_line, x='Year', y='Percentage with Electricity', title='Rate of Electricity in Schools (2013-2016)')

# # Display the line chart
# st.plotly_chart(fig)





#################

########################
# # State Wise Average Percentage of All Schools with Electricity
# state_wise_electricity = electricity.groupby('State_UT')['All Schools'].mean().sort_values(ascending=False)
# st.write("State Wise Average Percentage of All Schools with Electricity:")
# st.bar_chart(state_wise_electricity)
    #############################


# state_wise_electricity = electricity.groupby('State_UT')['All Schools'].mean()#.sort_values(ascending=False)

# st.header("State Wise Average Percentage of All Schools with Electricity")
# fig = px.funnel(data_frame=state_wise_electricity.reset_index(), x='State_UT', y='All Schools',
#                 labels={'State_UT': 'State/UT', 'All Schools': 'Average Percentage'},
#                 color_discrete_sequence=['teal'])

# st.plotly_chart(fig, use_container_width=False)
    

regions = {
        "North India": [
            'Haryana', 'Himachal Pradesh', 'Jammu And Kashmir', 'Punjab', 'Uttar Pradesh', 
             'Delhi', 'Chandigarh'
        ],
        "South India": [
            'Andhra Pradesh', 'Karnataka', 'Kerala', 'Tamil Nadu', 'Telangana', 
            'Puducherry', 'Lakshadweep'
        ],
        "East India": [
            'Bihar', 'Jharkhand', 'Odisha', 'West Bengal', 'Andaman & Nicobar Islands'
        ],
        "West India": [
            'Goa', 'Gujarat', 'Maharashtra', 'Rajasthan', 'Dadra & Nagar Haveli', 
            'Daman & Diu'
        ],
        "Central India": [
            'Chhattisgarh', 'Madhya Pradesh', 'MADHYA PRADESH'
        ],
        "North East India": [
            'Arunachal Pradesh', 'Assam', 'Manipur', 'Meghalaya', 'Mizoram', 
            'Nagaland', 'Sikkim', 'Tripura'
        ],
        "Union Territories": [
            'Pondicherry', 'Delhi', 'Chandigarh', 'Dadra & Nagar Haveli', 'Daman & Diu', 
            'Lakshadweep', 'Andaman & Nicobar Islands', 'Puducherry'
        ],
    
    }

# Create a state to region mapping
state_to_region = {state: region for region, states in regions.items() for state in states}

# Map each state to its region
electricity['Region'] = electricity['State_UT'].map(state_to_region)

# Order the regions in the DataFrame as per your defined order
ordered_regions = list(regions.keys())
electricity['Region'] = pd.Categorical(electricity['Region'], categories=ordered_regions, ordered=True)

# Group by region
region_wise_electricity = electricity.groupby('Region')['All Schools'].mean().reset_index()

# Create the funnel chart
fig = px.funnel(data_frame=region_wise_electricity, x='Region', y='All Schools',
                labels={'Region': 'Region', 'All Schools': 'Average Percentage'},
                color_discrete_sequence=['teal'])

# Display the chart
st.header("Region Wise Average Percentage of All Schools with Electricity")
st.plotly_chart(fig, use_container_width=True)
