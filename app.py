import pandas as pd
import pickle
from flask import Flask, render_template, request, jsonify

application = Flask(__name__)

# Load dataset
car = pd.read_csv("Cleaned_car.csv")

# Load trained ML model
model = pickle.load(open("LinearRegressionModel.pkl", "rb"))


@application.route('/')
def index():
    companies = sorted(car['company'].unique())
    return render_template('index.html', companies=companies, car_models=[], years=[], fuel_type=[],
                           predicted_price=None)


@application.route('/get_car_details', methods=['POST'])
def get_car_details():
    """Returns car details based on the selected company."""
    selected_company = request.json['company']

    filtered_cars = car[car['company'] == selected_company]

    car_models = sorted(filtered_cars['name'].unique().tolist())
    years = sorted(filtered_cars['year'].unique().tolist(), reverse=True)
    fuel_types = sorted(filtered_cars['fuel_type'].unique().tolist())

    return jsonify({'car_models': car_models, 'years': years, 'fuel_types': fuel_types})


@application.route('/predict', methods=['POST'])
def predict():
    company = request.form['company']
    name = request.form['name']
    year = int(request.form['year'])
    fuel_type = request.form['fuel_type']
    kms_driven = int(request.form['kms_driven'])

    # Prepare input for model
    input_data = pd.DataFrame([[company, name, year, fuel_type, kms_driven]],
                              columns=['company', 'name', 'year', 'fuel_type', 'kms_driven'])

    # Make prediction
    predicted_price = model.predict(input_data)[0]

    companies = sorted(car['company'].unique())

    return render_template('index.html', companies=companies, car_models=[], years=[], fuel_type=[],
                           predicted_price=round(predicted_price, 2))


if __name__ == "__main__":
    application.run(debug=True)
