import pandas as pd
import re
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ----------------- Model 1: car.py -----------------

# Load data for car.py
url = "https://raw.githubusercontent.com/farrasariffadhila/fp-rsbp-raw/refs/heads/main/modified_mmm.csv"
df_car = pd.read_csv(url)
df_car['harga'] = pd.to_numeric(df_car['harga'], errors='coerce')
df_car['jarak_tempuh'] = pd.to_numeric(df_car['jarak_tempuh'], errors='coerce')

# car.py functions
def extract_criteria(input_text):
    """Extract criteria from user input"""
    criteria = {
        'transmisi': None,
        'engine_size': None,
        'max_price': None,
        'tahun': None,
        'jarak_tempuh': None,
        'bahan_bakar': None,
        'max_pajak': None,
        'min_mpg': None
    }

    # Transmission
    if 'manual' in input_text.lower():
        criteria['transmisi'] = 'Manual'
    elif 'automatic' in input_text.lower():
        criteria['transmisi'] = 'Automatic'

    # Engine size
    match_engine = re.search(r'(\d+\.\d+|\d+)\s*(liter|cc|engine size)', input_text.lower())
    if match_engine:
        criteria['engine_size'] = float(match_engine.group(1))

    # Max price
    match_price = re.search(r'price\s*(under|less than|below)\s*(\d+)', input_text.lower())
    if match_price:
        criteria['max_price'] = int(match_price.group(2))

    # Year
    match_year = re.search(r'year\s*(\d{4})', input_text.lower())
    if match_year:
        criteria['tahun'] = int(match_year.group(1))

    # Mileage
    match_distance = re.search(r'(mileage|km|kilometers)\s*(under|less than|below)\s*(\d+)', input_text.lower())
    if match_distance:
        criteria['jarak_tempuh'] = int(match_distance.group(3))

    # Fuel type
    if 'diesel' in input_text.lower():
        criteria['bahan_bakar'] = 'Diesel'
    elif 'petrol' in input_text.lower() or 'gasoline' in input_text.lower():
        criteria['bahan_bakar'] = 'Petrol'

    # Tax 
    match_tax = re.search(r'tax\s*(under|less than|below)\s*(\d+)', input_text.lower())
    if match_tax:
        criteria['max_pajak'] = int(match_tax.group(2))

    # Minimum MPG 
    match_mpg = re.search(r'mpg\s*(at least|more than|greater than)\s*(\d+)', input_text.lower())
    if match_mpg:
        criteria['min_mpg'] = int(match_mpg.group(2))

    return criteria

def filter_cars(criteria):
    """Filter cars based on criteria"""
    filtered_cars = df_car.copy()

    if criteria['transmisi']:
        filtered_cars = filtered_cars[filtered_cars['transmisi'].str.contains(criteria['transmisi'], case=False, na=False)]
    if criteria['engine_size']:
        filtered_cars = filtered_cars[filtered_cars['ukuran_mesin'] == criteria['engine_size']]
    if criteria['max_price']:
        filtered_cars = filtered_cars[filtered_cars['harga'] <= criteria['max_price']]
    if criteria['tahun']:
        filtered_cars = filtered_cars[filtered_cars['tahun'] == criteria['tahun']]
    if criteria['jarak_tempuh']:
        filtered_cars = filtered_cars[filtered_cars['jarak_tempuh'] <= criteria['jarak_tempuh']]
    if criteria['bahan_bakar']:
        filtered_cars = filtered_cars[filtered_cars['bahan_bakar'].str.contains(criteria['bahan_bakar'], case=False, na=False)]
    if criteria['max_pajak']:
        filtered_cars = filtered_cars[filtered_cars['pajak'] <= criteria['max_pajak']]
    if criteria['min_mpg']:
        filtered_cars = filtered_cars[filtered_cars['mpg'] >= criteria['min_mpg']]

    return filtered_cars

@app.route('/api/get-cars', methods=['POST'])
def get_cars():
    """API Endpoint for Model 1"""
    data = request.get_json()
    user_input = data.get('query', '')

    # Extract criteria
    criteria = extract_criteria(user_input)

    # Print extracted criteria
    print("Criteria:", criteria)

    # utk pastikan kolom pajak tipenya numerik
    df_car['pajak'] = pd.to_numeric(df_car['pajak'], errors='coerce').fillna(0)
    filtered_cars = filter_cars(criteria)

    # cek hasil filter
    print("Filtered Data:", filtered_cars.head())

    if not filtered_cars.empty:
        filtered_cars['harga'] = filtered_cars['harga'].apply(lambda x: f"{x:,.0f}".replace(",", "."))
        filtered_cars['jarak_tempuh'] = filtered_cars['jarak_tempuh'].apply(lambda x: f"{x:,}".replace(",", "."))
        filtered_cars['pajak'] = filtered_cars['pajak'].apply(lambda x: f"{x:,.0f}".replace(",", "."))
        filtered_cars['mpg'] = filtered_cars['mpg'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "N/A")

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
    """API Endpoint for Model 2"""
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