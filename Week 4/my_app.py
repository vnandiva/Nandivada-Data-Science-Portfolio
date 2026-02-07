import streamlit as st
import pandas as pd
import os

st.title("My First EDA App")

# Interactive button
if st.button("Click me!"):
    st.write("🎉 You clicked the button!")

# Color picker
color = st.color_picker("Pick a color", "#00f900")
st.write(f"You picked: {color}")

# Load data from CSV
st.subheader("Data Explorer")
df = pd.read_csv("data/sample_data.csv")
# Show full dataset
st.write("Full dataset:")
st.dataframe(df)

# City filter
city = st.selectbox("Select a city", df["City"].unique())
filtered_df = df[df["City"] == city]

st.write(f"People in {city}:")
st.dataframe(filtered_df)


#add bar plot 
st.bar_chart(df['Salary'])

import seaborn as sns
boxplot1 = sns.boxplot(x=df['City'], y=df['Salary'])
st.pyplot(boxplot1.get_figure())


