import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression


st.set_page_config(layout="wide",
                   page_title="INDIAN SCHOOL EDUCATION STATISTICS ANALYSIS APP",
                   page_icon='📚')

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_option('deprecation.showPyplotGlobalUse', False)



rename_dict={"Primary_Total": "Primary", 
             "Upper Primary_Total": "Upper_Primary", 
             "Secondary _Total": "Secondary", 
             "HrSecondary_Total": "HrSecondary"}

rename_dict1={"Primary_Total": "Primary",
              "Upper_Primary_Total": "Upper_Primary",
              "Secondary_Total": "Secondary",
              "Higher_Secondary_Total": "HrSecondary"}
              


#df_enroll
# State_UT,Year,Primary_Boys,Primary_Girls,Primary_Total,Upper_Primary_Boys,Upper_Primary_Girls,Upper_Primary_Total,
# Secondary_Boys,Secondary_Girls,Secondary_Total,Higher_Secondary_Boys,Higher_Secondary_Girls,Higher_Secondary_Total

@st.cache_data()
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


    return df,india, 
 

with st.spinner("Processing Dropout data...."):
    df, india = load_data()

num_cols = df.select_dtypes(include='number').columns
cat_cols = df.select_dtypes(include='object').columns



st.markdown("""<h1 style="text-align: center;">Indian School Education Statistics Analysis</h1>""", unsafe_allow_html=True)
st.markdown("""
* **Python libraries:** pandas, streamlit, numpy, matplotlib, seaborn, geopandas
* **Data source:** [Indian School Education Statistics](https://data.gov.in/resources/stateut-wise-average-annual-drop-out-rate-2012-13-2014-15-ministry-human-resource
""")
c1, c2 = st.columns([3,2])
c1.image('https://mcmscache.epapr.in/post_images/website_350/post_29811255/full.jpg', use_column_width=True)




st.markdown("""<h1 style="text-align: center;">Data Analysis and Visualization</h1>""", unsafe_allow_html=True)
analysis_type = st.selectbox(
    "Select the type of analysis",
    options=["DROPOUT", "ENROLLMENT", "STATES WITH FACILITIES"]
)

if analysis_type == "DROPOUT":

    st.markdown("""<h2 style="text-align: center;">Dropout Analysis and Visualization</h2>""", unsafe_allow_html=True)
    states = df["State_UT"].unique()
    # st.write(states.tolist())
    c1, c2 = st.columns(2)
    c1.subheader("Select the state to visualize the dropout rates")
    state = c1.selectbox("State", states)
    col = c1.selectbox("Select the level to visualize the dropout rates", num_cols)
    df_state = df[df["State_UT"] == state]
    c1.dataframe(df_state, use_container_width=True)
    c2.subheader("Line chart of dropout rates for each level")


    fig = px.bar(
        data_frame=df_state,
        x="year",
        y=col,
        title=f"Dropout Rate for {col} Level in {state}",
        barmode="group",
        color_discrete_sequence=["#636efa", "#EF553B"],
        
        
    )

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title=f"Dropout Rate (%)",
        margin=dict(t=50, b=50, l=50, r=50),
        
    )

    # Display the plot using Streamlit
    c2.plotly_chart(fig, use_container_width=True)





    t1, t2 = st.tabs(["Bivariate","Trivariate"])
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
        

    # Create tabs
    tabs = st.tabs(["Best States", "Worst States","Gender Comparison"])

    # Best states tab
    with tabs[0]:
        st.subheader('Top 5 Best States in Dropout Rates')
        df_best5 = df.sort_values(by="Average", ascending=True).head(5)
        fig_best = px.pie(
            df_best5,
            values="Average",
            names="State_UT",
            color_discrete_sequence=["#ff9999", "#66b3ff", "#99ff99", "#99ffAA", "#ff6633"],
            hole=0.6,
        )

        # Add annotation and update layout
        fig_best.add_annotation(
            x=0,
            y=0,
            showarrow=False,
            font=dict(size=12, color="black"),
            xanchor="center",
            yanchor="middle",
        )
        fig_best.update_traces(textposition="inside", textinfo="percent+label")
        fig_best.update_layout(
            margin=dict(t=50, b=50, l=50, r=50),
            # title_x=0.5,
        )

        st.plotly_chart(fig_best, use_container_width=True)

    # Worst states tab
    with tabs[1]:
        st.subheader('Top 5 Worst States in Dropout Rates')
        df_worst5 = df.sort_values(by="Average", ascending=False).head(5)
        fig_worst = px.pie(
            df_worst5,
            values="Average",
            names="State_UT",
            color_discrete_sequence=["#ff9999", "#66b3ff", "#99ff99", "#99ffAA", "#ff6633"],
            hole=0.6,
        )

        # Add annotation and update layout
        fig_worst.add_annotation(
            x=0,
            y=0,
            text="Start",
            showarrow=False,
            font=dict(size=12, color="black"),
            xanchor="center",
            yanchor="middle",
        )
        fig_worst.update_traces(textposition="inside", textinfo="percent+label")
        fig_worst.update_layout(
            margin=dict(t=50, b=50, l=50, r=50),
            
        )

        st.plotly_chart(fig_worst, use_container_width=True)

    with tabs[2]:
        st.subheader('Comparison of Dropout Rates between Boys and Girls')
        df_boys=df[["Primary_Boys","Upper Primary_Boys","Secondary _Boys","HrSecondary_Boys"]]
        df_girls=df[["Primary_Girls","Upper Primary_Girls","Secondary _Girls","HrSecondary_Girls"]]
        df_boys = df_boys.apply(pd.to_numeric, errors='coerce')
        df_girls=df_girls.apply(pd.to_numeric, errors='coerce')
        boys_total = df_boys.sum().sum()
        girls_total = df_girls.sum().sum()
        boys_percentage = (boys_total / (boys_total + girls_total)) * 100
        girls_percentage = (girls_total / (boys_total + girls_total)) * 100
        labels = ['Boys', 'Girls']
        sizes = [boys_percentage, girls_percentage]
        colors = ['#ff9999', '#66b3ff']
        fig = go.Figure(data=[go.Pie(labels=labels, values=sizes, textinfo='label+percent',
                                marker=dict(colors=colors), hole=0.6)])
        st.plotly_chart(fig, use_container_width=True)

    df['year'] = pd.to_numeric(df['year'].str[:4], errors='coerce')
    state = st.selectbox('Select a State for Prediction', df['State_UT'].unique())
    education_level = st.selectbox('Select Education Level for Prediction', ['Primary', 'Upper_Primary', 'Secondary', 'HrSecondary'])
    state_data = df[df['State_UT'] == state]
    X = state_data[['year']].dropna()
    y = state_data[education_level].dropna()
    if len(X) != len(y):
        st.error('Insufficient data for prediction. Please try a different state or education level.')
    else:
        if st.button('Predict'):
            model = LinearRegression()
            model.fit(X, y)
            future_years = np.array([[year] for year in range(2016, 2021)]).reshape(-1, 1)
            predictions = model.predict(future_years)
            past_data = state_data[['year', education_level]].dropna()
            future_data = pd.DataFrame({'year': range(2016, 2021), education_level: predictions})
            plot_data = pd.concat([past_data, future_data])
            c1,c2 = st.columns(2)
            fig = px.line(plot_data, x='year', y=education_level, title=f'Predicted {education_level} Dropout Rates for {state}')
            fig.add_scatter(x=future_data['year'], y=future_data[education_level], mode='markers+lines', name='Predictions')
            c1.plotly_chart(fig)

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df_grouped = df.groupby("year")[num_cols].mean()
    merged = pd.merge(india, df, left_on="NAME_1", right_on="State_UT", how="inner")
    merged =merged[merged['NAME_1'] != 'Chandigarh']
    c1, c2 = st.columns(2)
    c1.subheader('Average Dropout Rates in India from 2012 to 2015')
    fig, ax = plt.subplots( figsize=(10, 9))
    ax.set_facecolor('#a0a64e')
    merged.plot(column='Average', cmap='YlOrBr', linewidth=1, ax=ax, edgecolor='0.5', legend=True )
    for idx, row in merged.iterrows():
        centroid_x, centroid_y = row['geometry'].centroid.x, row['geometry'].centroid.y
        state_name = row['NAME_1']
        ax.text(centroid_x, centroid_y, state_name, fontsize=10, ha='center', va='center', color='black')
    c1.pyplot(fig, use_container_width=True)

    c2.subheader('Dropout Rates Correlation Matrix')
    df_total = df[['Primary','Upper_Primary','Secondary','HrSecondary']]
    correlation_matrix = df_total.corr()
    fig,ax=plt.subplots(figsize=(10,8.5))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    c2.pyplot()

    # Calculate the total dropout rates for each level
    primary_total = df["Primary"].sum()
    upper_primary_total = df["Upper_Primary"].sum()
    secondary_total = df["Secondary"].sum()
    hrsecondary_total = df["HrSecondary"].sum()


    st.write("Total Dropout Rates:")
    st.write("Primary:", primary_total)
    st.write("Upper Primary:", upper_primary_total)
    st.write("Secondary:", secondary_total)
    st.write("Higher Secondary:", hrsecondary_total)
    
elif analysis_type == "ENROLLMENT":
    @st.cache_data()
    def load_data():
        df = pd.read_csv('gross-enrollment-ratio-2013-2016.csv')
        df['Year'] = df['Year'].apply(lambda x: int(x.split('-')[0]))  # Convert Year to integer

        

        return df
    

    with st.spinner("Processing Enrollment data...."):
        df = load_data()


    st.markdown("""<h2 style="text-align: center;">Enrollment Analysis and Visualization</h2>""", unsafe_allow_html=True)
    
    
    # Get unique states and education levels
    states = df["State_UT"].unique()
    education_levels = ['Primary', 'Upper_Primary', 'Secondary', 'Higher_Secondary']

    # Layout with two columns
    c1, c2 = st.columns(2)

    # User input in the left column
    c1.subheader("Select the State and Education Level")
    selected_state = c1.selectbox("State", states)
    selected_level = c1.selectbox("Education Level", education_levels)

    # Filter data based on selections
    df_state = df[df["State_UT"] == selected_state]

    # Display data in the left column
    c1.dataframe(df_state, use_container_width=True)

    # Visualization in the right column
    c2.subheader(f"Enrollment Rate for {selected_level} Level in {selected_state}")

    # Creating a bar plot
    fig = px.bar(
        data_frame=df_state,
        x="Year",
        y=f"{selected_level}_Total",
        # title=f"Enrollment Rate for {selected_level} Level in {selected_state}",
        barmode="group",
        color_discrete_sequence=[ "#EF553B"]
    )

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Enrollment Rate (%)",
        margin=dict(t=50, b=50, l=50, r=50)
    )

    # Display the plot using Streamlit
    c2.plotly_chart(fig, use_container_width=True)


    # State-wise Comparison
    st.header(f"State-wise Enrollment Comparison")
    # st.sidebar.header("User Input Parameters")
    selected_year = st.selectbox("Select Year", df['Year'].unique(), index=0)
    selected_level1 = st.selectbox("Select Education Level", ['Primary', 'Upper_Primary', 'Secondary', 'Higher_Secondary'], index=0)
    state_wise_data = df[df['Year'] == selected_year]

    # Creating a bar plot with increased size
    fig = px.bar(state_wise_data, x="State_UT", y=f"{selected_level1}_Total", 
                title=f"State-wise Enrollment in {selected_level1} Education ({selected_year})")

    # Update the layout to increase the size
    fig.update_layout(
        height=600,  # Height of the chart in pixels
        width=1200,   # Width of the chart in pixels
        xaxis_title="State/UT",
        yaxis_title="Enrollment Ratio",
        margin=dict(t=50, b=50, l=50, r=50),
        # polar=[ "#EF553B"],
    )

    # Display the plot using Streamlit
    st.plotly_chart(fig, use_container_width=False)  # Set to False to use the specified width


    c1,c2=st.columns(2)
    selected_state = c1.selectbox("Select State", df['State_UT'].unique(), index=0)
    selected_level = c1.selectbox("Select the Education Level", ['Primary', 'Upper_Primary', 'Secondary', 'Higher_Secondary'], index=0)

    # Gender Comparison
    c2.subheader(f"Gender Comparison in {selected_state} ({selected_year})")
    gender_data = df[(df['Year'] == selected_year) & (df['State_UT'] == selected_state)]
    fig = go.Figure(data=[go.Pie(labels=['Boys', 'Girls'], values=[gender_data[f"{selected_level}_Boys"].values[0], gender_data[f"{selected_level}_Girls"].values[0]])])
    c2.plotly_chart(fig, use_container_width=True)


    
    if df['Year'].dtype == 'O':  # 'O' stands for object, typically used for strings in pandas
        df['Year'] = pd.to_numeric(df['Year'].str[:4], errors='coerce')

    # User selects a state and education level
    state = st.selectbox('Select a State for Prediction', df['State_UT'].unique())
    education_level = st.selectbox('Select Education Level for Prediction', ['Primary_Total', 'Upper_Primary_Total', 'Secondary_Total', 'Higher_Secondary_Total'])

    # Filter data based on the state
    state_data = df[df['State_UT'] == state]

    # Prepare data for model
    X = state_data[['Year']].dropna()
    y = state_data[education_level].dropna()

    # Check if data is sufficient for prediction
    if len(X) != len(y):
        st.error('Insufficient data for prediction. Please try a different state or education level.')
    else:
        # Predict future values when the button is clicked
        if st.button('Predict'):
            model = LinearRegression()
            model.fit(X, y)
            future_years = np.array([[year] for year in range(2017, 2022)]).reshape(-1, 1)
            predictions = model.predict(future_years)

            # Combine past data and future predictions
            past_data = state_data[['Year', education_level]].dropna()
            future_data = pd.DataFrame({'Year': range(2017, 2022), education_level: predictions})
            plot_data = pd.concat([past_data, future_data])

            # Plotting
            c1, c2 = st.columns(2)
            fig = px.line(plot_data, x='Year', y=education_level, title=f'Predicted {education_level} Enrollment Rates for {state}')
            fig.add_scatter(x=future_data['Year'], y=future_data[education_level], mode='markers+lines', name='Predictions')
            c1.plotly_chart(fig, use_container_width=True)


        regions = {
        "North India": [
            'Haryana', 'Himachal Pradesh', 'Jammu And Kashmir', 'Punjab', 'Uttar Pradesh', 
            'Uttarakhand', 'Delhi', 'Chandigarh'
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

    # Streamlit UI
    st.header('Region wise Enrollment Data Visualization')

    # Dropdown for selecting the region
    selected_region = st.selectbox("Select Region", list(regions.keys()))

    # Dropdown for selecting the education level
    education_levels = ['Primary_Total', 'Upper_Primary_Total', 'Secondary_Total', 'Higher_Secondary_Total']  # Modify as per your column names
    selected_level = st.selectbox("Select Education Level", education_levels)

    # Filter data based on the selected region and education level
    filtered_df = df[df['State_UT'].isin(regions[selected_region])]
    filtered_df = filtered_df[filtered_df['Year'].isin([2013, 2014, 2015])]

    # Pivot the DataFrame
    pivot_df = filtered_df.pivot(index="State_UT", columns="Year", values=selected_level).reset_index()

    # Prepare data for Plotly
    plotly_data = pivot_df.melt(id_vars=['State_UT'], var_name='Year', value_name=selected_level)

    # Create the heatmap with Plotly
    fig = px.imshow(plotly_data.pivot(index='State_UT', columns='Year', values=selected_level),
                    labels=dict(x="Year", y="State/UT", color=selected_level),
                    x=plotly_data['Year'].unique(),
                    y=plotly_data['State_UT'].unique(),
                    title=f'Enrollment Rate Heatmap for {selected_level} in {selected_region}',
                    color_continuous_scale='Viridis'
                    )

    # Update the layout
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=[2013, 2014, 2015],
            ticktext=['2013', '2014', '2015']
        )
    )

    # Show the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)







