import streamlit as st
import plotly.express as px
import pandas as pd
import joblib


'''
# Our Last Morning Kick-Off :sob: :sob: :sob: :sob: :sob:

You can write in Markdown *just by typing like this!*

For all your Streamlit questions, check out their [documentation](https://docs.streamlit.io/en/stable/api.html)
'''


'''
## I. Plot a bar chart with COVID-19 Cases dataset :mask: :mask: :mask:
'''

# Covid Cases Visualization 
covid = pd.read_csv('data/confirmed_covid19_cases.csv')

# You can use this direct link to load the csv file as well: "https://raw.githubusercontent.com/shinokada/covid-19-stats/master/data/daily-new-confirmed-cases-of-covid-19-tests-per-case.csv"

# Change the column names
covid.columns = ['Country', 'Code', 'Date', 'Confirmed', 'Days since confirmed']

# Change the date format
covid['Date'] = pd.to_datetime(covid['Date']).dt.strftime('%Y-%m-%d')

# Show the covid dataframe in Streamlit
st.write(covid)

# Create unique values in the column "Country" and "Date"
country_options = covid['Country'].unique().tolist()
date_options = covid['Date'].unique().tolist()

# Create select box for date
date = st.selectbox('Which date would you like to see?', date_options, 100)

# Create select box for country
country = st.multiselect('Which country would you like to see?', country_options, ['Brazil'])

# Assign new values for variable 'covid' for visualization
covid = covid[covid['Country'].isin(country)]

#covid = covid[covid['Date'] == date] <= use this code if you don't want to use the buttons for Date and delete the 'animation_frame' & 'animation_group' parameter below

# Plot bar chart
fig = px.bar(covid, x= 'Country', y= 'Confirmed', color= 'Country', title='Global COVID-19 Cases',
				range_y=[0, 35000], animation_frame='Date', animation_group='Country')

# Adjust the speed of the graph
fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 30
fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 5

fig.update_layout(width=800)

# Show the plot
st.write(fig)


'''
## II. Building predictive models :mag: :mag: :mag:
'''


@st.cache(allow_output_mutation=True)

def load_model(path):
    model = joblib.load(path)
    return model

# Load the model
model = load_model('data/sentiment_pipeline.pkl')

# Create text box:
text = st.text_input('Enter your review text below', 'BrainStation is awesome')

# Create variable 'prediction' to get the input from "text" variable and produce the output for the sentiment prediction
prediction = model.predict({text})
if prediction == 1:
    'We predict that this is a positive review!'
else:
    'We predict that this is a negative review!'



