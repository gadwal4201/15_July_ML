from flask import Flask, request, render_template
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the pre-trained model and scaler
ANN_model = load_model('./models/loan_approval.h5')
std_scaler = joblib.load('./models/std_scaler.lb')

# Define expected columns for the model
expected_columns = ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term',
                     'cibil_score', 'residential_assets_value', 'commercial_assets_value',
                     'luxury_assets_value', 'bank_asset_value', 'education_Not Graduate',
                     'self_employed_Yes']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Fetching data from the form
        input_data = {
            'no_of_dependents': int(request.form['no_of_dependents']),
            'income_annum': float(request.form['income_annum']),
            'loan_amount': float(request.form['loan_amount']),
            'loan_term': float(request.form['loan_term']),
            'cibil_score': float(request.form['cibil_score']),
            'residential_assets_value': float(request.form['residential_assets_value']),
            'commercial_assets_value': float(request.form['commercial_assets_value']),
            'luxury_assets_value': float(request.form['luxury_assets_value']),
            'bank_asset_value': float(request.form['bank_asset_value']),
            'education_Not Graduate': int(request.form['education_Not Graduate']),
            'self_employed_Yes': int(request.form['self_employed_Yes'])
        }

        # Create a DataFrame for the input data
        input_df = pd.DataFrame([input_data])

        # Ensure that all expected columns are present
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder columns to match the training schema
        input_df = input_df[expected_columns]

        # Preprocess the input using the same scaler used during training
        scaled_data = std_scaler.transform(input_df)

        # Make the prediction
        prediction = ANN_model.predict(scaled_data)
        prediction_class = (prediction > 0.5).astype(int)

        # Return result
        if prediction_class[0][0] == 0:
            result = "Your loan application is likely to be approved."
        else:
            result = "Your loan application is likely to be rejected."

        return render_template('output.html', prediction_text=result)

    except Exception as e:
        return f"An error occurred during prediction: {e}"

if __name__ == "__main__":
    app.run(debug=True)
