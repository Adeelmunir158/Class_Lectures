import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import wbdata
import pandas as pd
import plotly.express as px
import webbrowser
import threading

# Load the data
# Define the indicator for population (SP.POP.TOTL)
indicator = {
    'SP.POP.TOTL': 'total_population',  # Population 
    'SM.POP.NETM': 'net_migration'     # Net Migration
}
# Define countries (India, Pakistan, Bangladesh, Sri Lanka, Afghanistan)
# countries = ['IN', 'PK', 'BD', 'LK', 'AF']  # ISO codes
# Fetch data
data = wbdata.get_dataframe(indicator, 
                            # country=countries
                            )
# Reset index to convert it into a DataFrame
data.reset_index(inplace=True)
# Rename columns for clarity
data.rename(columns={'country': 'Country', 'date': 'Year'}, inplace=True)
# Ensure Year column is numeric
data['Year'] = pd.to_numeric(data['Year'])
# Filter data between 1960 and 2023
data = data[(data['Year'] >= 1960) & (data['Year'] <= 2023)]

# Create a DataFrame
df = data.copy()
# initialize the Dash app
app = dash.Dash(__name__)