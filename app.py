import pandas as pd
import re
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.pipeline import Pipeline

app = Flask(__name__)
CORS(app)

# ----------------- Model 1: car.py -----------------

# Load data for car.py
url = "https://raw.githubusercontent.com/farrasariffadhila/fp-rsbp-raw/refs/heads/main/modified_mmm.csv"
df_car = pd.read_csv(url)
df_car['harga'] = pd.to_numeric(df_car['harga'], errors='coerce')

# car.py functions
def extract_criteria(input_text):
    transmisi, engine_size, max_price = None, None, None

    if 'manual' in input_text.lower():
        transmisi = 'Manual'
    elif 'automatic' in input_text.lower():
        transmisi = 'Automatic'

    match_engine = re.search(r'(\d+\.\d+|\d+)\s*(liter|cc|engine size)', input_text.lower())
    if match_engine:
        engine_size = float(match_engine.group(1))

    match_price = re.search(r'under\s*(\d+)', input_text.lower())
    if match_price:
        max_price = int(match_price.group(1))

    return transmisi, engine_size, max_price

def filter_cars(transmisi, engine_size, max_price):
    filtered_cars = df_car.copy()
    if transmisi:
        filtered_cars = filtered_cars[filtered_cars['transmisi'].str.contains(transmisi, case=False, na=False)]
    if engine_size:
        filtered_cars = filtered_cars[filtered_cars['ukuran_mesin'] == engine_size]
    if max_price:
        filtered_cars = filtered_cars[filtered_cars['harga'] <= max_price]
    return filtered_cars

@app.route('/api/get-cars', methods=['POST'])
def get_cars():
    data = request.get_json()
    user_input = data.get('query', '')
    transmisi, engine_size, max_price = extract_criteria(user_input)
    filtered_cars = filter_cars(transmisi, engine_size, max_price)

    if not filtered_cars.empty:
        return jsonify({'cars': filtered_cars.to_dict(orient="records")}), 200
    return jsonify({'message': 'No cars found with the given criteria.'}), 200

# ----------------- Model 2: new-model.py -----------------

# Load data and model for new-model.py
df_new_model = pd.read_csv("dataset.csv")
with open("knn_model.pkl", "rb") as file:
    knn_model = pickle.load(file)

def recommend_car_by_user_input(user_input, df, pipeline, n_recommendations=5):
    for col in ["Merek", "Model"]:
        if not user_input[col]:
            user_input[col] = df[col].mode()[0]
    user_input_df = pd.DataFrame([user_input])
    preprocessed_input = pipeline.named_steps["preprocessor"].transform(user_input_df)
    distances, indices = pipeline.named_steps["knn"].kneighbors(preprocessed_input, n_neighbors=n_recommendations)
    return df.iloc[indices[0]]

@app.route('/api/get-new-cars', methods=['POST'])
def get_new_cars():
    data = request.get_json()
    try:
        user_preferences = {
            "Merek": data.get("brand", ""),
            "Model": data.get("model", ""),
            "harga": float(data.get("price", 0)),
            "Jarak tempuh": float(data.get("mileage", 0)),
            "Tipe bahan bakar": data.get("fuel", ""),
            "Transmisi": data.get("transmission", ""),
            "Kapasitas mesin": float(data.get("engine", 0)),
            "Tahun": float(data.get("year", 0)),
        }
        recommendations = recommend_car_by_user_input(user_preferences, df_new_model, knn_model)
        return jsonify({'cars': recommendations.to_dict(orient="records")}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ----------------- Run Flask Server -----------------
if __name__ == "__main__":
    app.run(debug=True)
