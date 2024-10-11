import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics

# Load the dataset
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# Format the data for Prophet
df_train = train_data[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
df_test = test_data[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})

# Define a function to tune hyperparameters
def tune_prophet(df_train, param_grid):
    best_rmse = float("inf")
    best_params = None

    for params in param_grid:
        print(f"Training with parameters: {params}")
        
        # Initialize Prophet model with custom parameters
        model = Prophet(
            changepoint_prior_scale=params['changepoint_prior_scale'],
            seasonality_prior_scale=params['seasonality_prior_scale'],
            holidays_prior_scale=params['holidays_prior_scale'],
            yearly_seasonality=params['yearly_seasonality'],
            weekly_seasonality=params['weekly_seasonality'],
            daily_seasonality=params['daily_seasonality']
        )
        
        # Fit the model
        model.fit(df_train)
        
        # Cross-validation
        df_cv = cross_validation(model, initial='365 days', period='180 days', horizon='365 days')
        
        # Calculate RMSE for the model
        df_p = performance_metrics(df_cv)
        rmse = df_p['rmse'].values[0]
        
        # Check if this is the best model so far
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params

    return best_params, best_rmse

# Define the parameter grid for tuning
param_grid = [
    {
        'changepoint_prior_scale': [0.001, 0.01, 0.05],
        'seasonality_prior_scale': [0.01, 0.1, 1.0],
        'holidays_prior_scale': [0.01, 0.1],
        'yearly_seasonality': [True, False],
        'weekly_seasonality': [True, False],
        'daily_seasonality': [False]  # Tuning mainly yearly/weekly, usually daily is not needed for demand
    }
]

# Tune hyperparameters and find the best model
best_params, best_rmse = tune_prophet(df_train, param_grid)
print(f"Best Parameters: {best_params}")
print(f"Best RMSE from Cross-validation: {best_rmse}")

# Step 5: Train the final Prophet model using the best hyperparameters
model = Prophet(
    changepoint_prior_scale=best_params['changepoint_prior_scale'],
    seasonality_prior_scale=best_params['seasonality_prior_scale'],
    holidays_prior_scale=best_params['holidays_prior_scale'],
    yearly_seasonality=best_params['yearly_seasonality'],
    weekly_seasonality=best_params['weekly_seasonality'],
    daily_seasonality=best_params['daily_seasonality']
)

model.fit(df_train)

# Make predictions on the test set
forecast = model.predict(df_test[['ds']])

# Evaluate the final model using RMSE and MAE
rmse = mean_squared_error(df_test['y'], forecast['yhat'], squared=False)
mae = mean_absolute_error(df_test['y'], forecast['yhat'])
print(f"Final Prophet Model - RMSE: {rmse}, MAE: {mae}")

