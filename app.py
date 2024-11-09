from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Load the model
MODEL_PATH = "final_logistic_regression_model.pkl"
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# Home route with form
@app.route("/")
def home():
    today = datetime.now().date()
    return render_template("index.html", today=today)  # Pass current date to the template

# Prediction route to handle form data
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract feature values from form data
        recency_date = request.form.get("Recency(Date)")  # Last visit date
        recency_months = float(request.form.get("Recency(Months)"))  # Months since last visit
        frequency = float(request.form.get("Frequency"))  # Number of visits in last year
        health_status = int(request.form.get("Class"))  # Health risk status (0 or 1)
        time = float(request.form.get("Time"))  # Time since diagnosis/treatment
        
        # If recency date is provided, calculate the number of months since the last visit
        if recency_date:
            last_visit_date = datetime.strptime(recency_date, "%Y-%m-%d")
            current_date = datetime.now()
            delta_months = (current_date.year - last_visit_date.year) * 12 + current_date.month - last_visit_date.month
            recency = delta_months  # Use calculated months since the last visit
        else:
            recency = recency_months  # Use the user-provided months if no date is provided
        
        # Automatically flag as high risk if the recency is greater than 12 months (1 year)
        if recency > 12:
            prediction_message = "Yes, you have a health risk (long time since last visit)."
            return render_template("index.html", prediction_message=prediction_message, today=datetime.now().date())
        
        # Create feature array with 4 features
        features = [recency, frequency, health_status, time]
        
        # Reshape and predict
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)
        
        # Map prediction to readable message
        if prediction[0] == 0:
            prediction_message = "No, you do not have a health risk."
        else:
            prediction_message = "Yes, you have a health risk."
        
        # Return prediction to the template
        return render_template("index.html", prediction_message=prediction_message, today=datetime.now().date())
    
    except Exception as e:
        # Return an error message if something goes wrong
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
