import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet import Prophet
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load preprocessed data
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# Prophet Model for Forecasting
def prophet_forecast(train_data, test_data):
    df_train = train_data[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
    df_test = test_data[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
    
    model = Prophet()
    model.fit(df_train)
    
    forecast = model.predict(df_test[['ds']])
    
    # Evaluate
    rmse = mean_squared_error(df_test['y'], forecast['yhat'], squared=False)
    mae = mean_absolute_error(df_test['y'], forecast['yhat'])
    
    print(f"Prophet RMSE: {rmse}, MAE: {mae}")
    return forecast

# LightGBM Model for Forecasting
def lightgbm_forecast(train_data, test_data):
    X_train = train_data.drop(['sales'], axis=1)
    y_train = train_data['sales']
    X_test = test_data.drop(['sales'], axis=1)
    y_test = test_data['sales']

    model = lgb.LGBMRegressor()
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    # Evaluate
    rmse = mean_squared_error(y_test, predictions, squared=False)
    mae = mean_absolute_error(y_test, predictions)
    
    print(f"LightGBM RMSE: {rmse}, MAE: {mae}")
    return predictions

# LSTM Model for Time-Series Forecasting
def lstm_forecast(train_data, test_data):
    # Reshaping data for LSTM [samples, time_steps, features]
    X_train = train_data.drop(['sales'], axis=1).values.reshape(-1, 3, train_data.shape[1]-1)  # 3 lag features
    y_train = train_data['sales'].values
    X_test = test_data.drop(['sales'], axis=1).values.reshape(-1, 3, test_data.shape[1]-1)
    y_test = test_data['sales'].values

    # LSTM Model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate
    rmse = mean_squared_error(y_test, predictions, squared=False)
    mae = mean_absolute_error(y_test, predictions)
    
    print(f"LSTM RMSE: {rmse}, MAE: {mae}")
    return predictions

# Run the models
print("Prophet Forecasting:")
prophet_forecast(train_data, test_data)

print("\nLightGBM Forecasting:")
lightgbm_forecast(train_data, test_data)

print("\nLSTM Forecasting:")
lstm_forecast(train_data, test_data)
