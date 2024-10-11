from flask import Flask, request, jsonify
import pandas as pd
import lightgbm as lgb

app = Flask(__name__)

# Load the pre-trained model
model = lgb.Booster(model_file='lightgbm_demand_model.txt')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame(data)
    
    # Make prediction
    prediction = model.predict(df)
    
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
