import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go


st.set_page_config(layout="wide",
                   page_title="INDIAN SCHOOL EDUCATION STATISTICS ANALYSIS APP",
                   page_icon='📚')

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_option('deprecation.showPyplotGlobalUse', False)



rename_dict={"Primary_Total": "Primary", 
             "Upper Primary_Total": "Upper_Primary", 
             "Secondary _Total": "Secondary", 
             "HrSecondary_Total": "HrSecondary"}



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
    return df, india
 

with st.spinner("Processing Immigration data...."):
    df, india = load_data()

num_cols = df.select_dtypes(include='number').columns
cat_cols = df.select_dtypes(include='object').columns


c1, c2 = st.columns([3,2])
c1.image('https://mcmscache.epapr.in/post_images/website_350/post_29811255/full.jpg', use_column_width=True)
st.title('Indian School Education Statistics Analysis')
st.markdown("""
* **Python libraries:** pandas, streamlit, numpy, matplotlib, seaborn, geopandas
* **Data source:** [Indian School Education Statistics](https://data.gov.in/resources/stateut-wise-average-annual-drop-out-rate-2012-13-2014-15-ministry-human-resource
""")


c2.info("Raw dataset")
c2.dataframe(df, use_container_width=True)

st.markdown("""<h1 style="text-align: center;">Dropout Analysis and Visualization</h1>""", unsafe_allow_html=True)

states = df["State_UT"].unique()
# st.write(states.tolist())
c1, c2 = st.columns(2)
c1.subheader("Select the state to visualize the dropout rates")
state = c1.selectbox("State", states)
df_state = df[df["State_UT"] == state]
c1.dataframe(df_state, use_container_width=True)
col = c1.selectbox("Select the level to visualize the dropout rates", num_cols)
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
    # Calculate the total dropout rates for boys and girls
    boys_total = df_boys.sum().sum()
    girls_total = df_girls.sum().sum()

    # Calculate the percentage of dropout rates for boys and girls
    boys_percentage = (boys_total / (boys_total + girls_total)) * 100
    girls_percentage = (girls_total / (boys_total + girls_total)) * 100
    # Create a pie chart to visualize the comparison
    labels = ['Boys', 'Girls']
    sizes = [boys_percentage, girls_percentage]
    colors = ['#ff9999', '#66b3ff']

    fig = go.Figure(data=[go.Pie(labels=labels, values=sizes, textinfo='label+percent',
                             marker=dict(colors=colors), hole=0.6)])
    st.plotly_chart(fig, use_container_width=True)


df["year"] = pd.to_numeric(df["year"], errors="coerce")
df_grouped = df.groupby("year")[num_cols].mean().reset_index()
merged = pd.merge(india, df, left_on="NAME_1", right_on="State_UT", how="inner")
merged =merged[merged['NAME_1'] != 'Chandigarh']




c1, c2 = st.columns(2)
c1.subheader('Average Dropout Rates in India from 2012 to 2015')
fig, ax = plt.subplots( figsize=(10, 9))
ax.set_facecolor('#a0a64e')

  # Increase the figsize to increase the size of the map
merged.plot(column='Average', cmap='YlOrBr', linewidth=1, ax=ax, edgecolor='0.5', legend=True )
for idx, row in merged.iterrows():
    centroid_x, centroid_y = row['geometry'].centroid.x, row['geometry'].centroid.y
    state_name = row['NAME_1']
    ax.text(centroid_x, centroid_y, state_name, fontsize=10, ha='center', va='center', color='black')
c1.pyplot(fig, use_container_width=True)

c2.subheader('Dropout Rates Correlation Matrix')
df_total = df[['Primary','Upper_Primary','Secondary','HrSecondary']]
fig,ax=plt.subplots(figsize=(10,8.5))
sns.heatmap(df_total.corr(), annot=True, cmap="Blues", ax=ax)
#ax.set_title("Correlation matrix of dropout rates for each level")
c2.pyplot(fig, use_container_width=True)


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













