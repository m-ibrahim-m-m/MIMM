import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
import yfinance as yf
from keras.models import load_model
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import seaborn as sns
from datetime import datetime
from sklearn.metrics import mean_squared_error


st.title('Stock trend prediction')

user_input = st.text_input('Enter Stock Ticker AMOC.CA HRHO.CA ETEL.CA COMI.CA CCAP.CA','ETEL.CA')


# Download stock data for the last 24 years (for example)
data = yf.download(user_input, start='2000-01-01', end = '2024-12-22')

#Discribing the data
st.subheader('Data from 2000 - 2024')
st.write(data.describe())


#visualiztion
st.subheader('Closing price VS Time Chart')
fig1 = plt.figure(figsize=(12, 6))
plt.plot(data.Close)
plt.title(f'{user_input} data')
plt.xlabel('Year')
plt.ylabel('Closing Price')
st.pyplot(fig1)


st.subheader('Moving Average')
ma20 = data.Close.rolling(20).mean()
fig2 = plt.figure(figsize=(12, 6))
plt.plot(ma20,'y')
plt.plot(data.Close,'b')
ma50 = data.Close.rolling(50).mean()
plt.plot(ma50,'g')
ma100 = data.Close.rolling(100).mean()
plt.plot(ma100,'r')
plt.title(f'{user_input} data')
plt.xlabel('Year')
plt.ylabel('Moving Average')
plt.legend()
st.pyplot(fig2)


# Download historical data for Apple stock (AAPL) for one year
data1 = yf.download(user_input, start='2023-01-01', end='2024-01-01')
      
# Calculate the daily return
data1['Daily Return'] = data1['Close'].pct_change()

# Calculate the average daily return
average_daily_return = data1['Daily Return'].mean()

# Create a formatted string for the result
Daily_Return = f"The average daily return for the {user_input} stock is: {average_daily_return * 100:.2f}%"

# Display the result in Streamlit
st.subheader('Daily Return')
st.write(Daily_Return)

#Scaling the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.values)



st.subheader('Log Returns')
data['Log Returns'] = np.log(data['Close'] / data['Close'].shift(1))
fig5 = plt.figure(figsize=(12, 6))
plt.plot(data['Log Returns'], label=f'{user_input} Log Returns', color='green')
plt.title(f'{user_input} Log Returns')
plt.xlabel('Date')
plt.ylabel('Log Return')
plt.legend()
plt.grid(True)
plt.show()
st.pyplot(fig5)


st.subheader('Log Returns Distribution')
# Plot histogram of returns
Fig4 = plt.figure(figsize=(12, 6))
plt.hist(data['Log Returns'].dropna(), bins=50, color='purple', edgecolor='black', alpha=0.7)
plt.title(f'{user_input} Returns Distribution')
plt.xlabel('Log Return')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
st.pyplot(Fig4)


# Scaling the 'Close' column only
close_data = data[['Close']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_data)

# Function to create dataset for LSTM (X_train, y_train) based on time steps
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Prepare datasets
time_step = 60
X, y = create_dataset(scaled_data, time_step)

# Reshaping data for LSTM input: [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Loading the Model
model = load_model('my_model1.keras')

# Split the data into training and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Predict the stock prices for the test set
predicted_stock_price = model.predict(X_test)

st.subheader('Model Perdiction Error')
rmse = np.sqrt(mean_squared_error(predicted_stock_price,y_test))
st.write(f'Root Mean Squared Error percentage for the Perdiction Model is {rmse* 100:.2f}%')

# Inverse transform the predictions and actual values to get original scale
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plotting the results
st.subheader('Perdiction Vs Actual price')
fig3 = plt.figure(figsize=(12,6))
plt.plot(y_test, color='blue', label=f'{user_input} Actual Price')
plt.plot(predicted_stock_price, color='red', label=f'{user_input} Predicted Price')
plt.title(f'{user_input} Stock Price Prediction vs Actual')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
st.pyplot(fig3)




rets = data.dropna()

# Calculate the mean and standard deviation for each column (e.g., asset)
mean_returns = rets.mean()
std_returns = rets.std()

# Calculate the area for the scatter plot (optional: you can adjust this logic as needed)
area = np.pi * (20 ** 2)  # Adjust size if necessary

# Create the plot
fig6 = plt.figure(figsize=(12, 6))

# Scatter plot
plt.scatter(mean_returns, std_returns, s=area)

# Labels
plt.xlabel('Expected return')
plt.ylabel('Risk')
plt.title(f'How much value do we put at risk by investing in {user_input} stock')

# Annotation: Make sure `user_input` is defined before this
plt.annotate(user_input, xy=(mean_returns[0], std_returns[0]), xytext=(50, 50), 
             textcoords='offset points', ha='right', va='bottom', 
             arrowprops=dict(arrowstyle='-', color='blue', connectionstyle='arc3,rad=-0.3'))

plt.show()
st.pyplot(fig6)
