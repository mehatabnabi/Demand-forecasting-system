# Demand Forecasting System with Prophet Hyperparameter Tuning

This project builds an end-to-end demand forecasting system using **Prophet** with hyperparameter tuning, along with **LightGBM** and **LSTM** models. The models predict future product demand based on historical sales data and are evaluated using **RMSE** and **MAE**.

## Key Features
- **Models Used:** Prophet, LightGBM, LSTM.
- **Evaluation Metrics:** RMSE and MAE.
- **Hyperparameter Tuning:** Grid search is used to tune Prophet's key parameters such as `changepoint_prior_scale`, `seasonality_prior_scale`, and `holidays_prior_scale`.
- **Tech:** Python, Pandas, NumPy, LightGBM, TensorFlow/Keras, Prophet.

## Project Structure
- **data_preprocessing.py:** Script to clean and prepare the dataset.
- **demand_forecasting.py:** Script to build and evaluate different forecasting models (Prophet, LightGBM, and LSTM).
- **prophet_tuning.py:** Script to perform hyperparameter tuning for the Prophet model.

## How to Run
Install the required dependencies: `pip install -r requirements.txt`
Run `data_preprocessing.py` to preprocess the dataset.
Run `prophet_tuning.py` to tune and train the Prophet model.
Run `demand_forecasting.py` to train and evaluate the other models.

## Results
- The **RMSE** and **MAE** scores for each model will be the outputs.
