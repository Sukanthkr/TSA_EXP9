# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 
### Date: 23/10/2025
### Name:SUKANTH KR
### Reg:212224040338

### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load the data
data = pd.read_csv("gold_price_data.csv")

# Convert 'Date' column to datetime and set it as index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Define the ARIMA model function
def arima_model(data, target_variable, order):
    # Split data into training and testing sets
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    # Fit the ARIMA model
    model = ARIMA(train_data[target_variable], order=order)
    fitted_model = model.fit()

    # Make predictions
    forecast = fitted_model.forecast(steps=len(test_data))

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(test_data[target_variable], forecast))

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data[target_variable], label='Training Data')
    plt.plot(test_data.index, test_data[target_variable], label='Testing Data')
    plt.plot(test_data.index, forecast, label='Forecasted Data')
    plt.xlabel('Date')
    plt.ylabel(target_variable)
    plt.title('ARIMA Forecasting for ' + target_variable)
    plt.legend()
    plt.show()

    print("Root Mean Squared Error (RMSE):", rmse)

# Run the ARIMA model on the 'Value' column
arima_model(data, 'Value', order=(5,1,0))
```
### OUTPUT:

<img width="859" height="547" alt="download" src="https://github.com/user-attachments/assets/58d9986f-c3fd-4baf-8a09-f4db61c44843" />

<img width="495" height="35" alt="image" src="https://github.com/user-attachments/assets/90411220-024d-4d1f-a490-c8b8d943cf0f" />

### RESULT:
Thus the program run successfully based on the ARIMA model using python.
