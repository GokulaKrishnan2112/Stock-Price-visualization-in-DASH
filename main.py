#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html

# Function to load stock data
def load_data(ticker):
    data = yf.download(ticker, start="2010-01-01", end="2021-12-31")
    return data

# Function to split the data into training and testing sets
def split_data(data, test_size=0.2):
    return train_test_split(data, test_size=test_size)

# Function to fit an ARIMA model and make predictions
def predict_price(train, test):
    model = ARIMA(train, order=(5, 1, 0))
    model_fit = model.fit()
    prediction = model_fit.forecast(steps=len(test))
    return prediction

# Load the data
data = load_data("AAPL")

# Split the data into training and testing sets
train, test = split_data(data["Close"], test_size=0.2)

# Reset the index of the test set to a continuous range of numbers
test = test.reset_index(drop=True)

# Predict the stock price
prediction = predict_price(train, test)

# Initialize the dash app
app = dash.Dash()

# Create the layout for the dashboard
app.layout = html.Div([
    html.H1("Stock Price Prediction"),
    html.Div([
        html.Label("Enter the stock ticker:"),
        dcc.Input(id="ticker", value="AAPL", type="text"),
        html.Button(id="submit", n_clicks=0, children="Submit"),
    ]),
    html.Div(id="output"),
])

# Update the data when the submit button is clicked
@app.callback(
    dash.dependencies.Output(component_id="output", component_property="children"),
    [dash.dependencies.Input(component_id="submit", component_property="n_clicks")],
    [dash.dependencies.State(component_id="ticker", component_property="value")],
)
def update_data(n_clicks, ticker):
    data = load_data(ticker)
    train, test = split_data(data["Close"], test_size=0.2)
    test = test.reset_index(drop=True)
    prediction = predict_price(train, test)

    return html.Div([
        html.H2("Stock Price Forecast"),
        dcc.Graph(figure=px.line(x=test.index, y=prediction)),
    ])

if __name__ == '__main__':
    app.run_server(debug=True)


# In[ ]:




