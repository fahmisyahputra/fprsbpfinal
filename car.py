import pandas as pd
import re
from flask import Flask, request, jsonify
from flask_cors import CORS

# Membaca data dari URL
url = "https://raw.githubusercontent.com/farrasariffadhila/fp-rsbp-raw/refs/heads/main/modified_mmm.csv"
df = pd.read_csv(url)

# Pastikan kolom harga adalah numerik
df['harga'] = pd.to_numeric(df['harga'], errors='coerce')

# Fungsi untuk mengekstrak kriteria dari input
def extract_criteria(input_text):
    transmisi = None
    engine_size = None
    max_price = None

    # Menentukan transmisi
    if 'manual' in input_text.lower() or 'manual transmission' in input_text.lower():
        transmisi = 'Manual'
    elif 'automatic' in input_text.lower() or 'automatic transmission' in input_text.lower():
        transmisi = 'Automatic'

    # Menentukan ukuran mesin dengan regex yang lebih fleksibel
    match_engine = re.search(r'(\d+\.\d+|\d+)\s*(liter|engine size|ukuran mesin|cc|kapasitas)', input_text.lower())
    if match_engine:
        try:
            engine_size = float(match_engine.group(1))  # Ambil angka sebelum "liter" atau "engine size"
        except (IndexError, TypeError):
            engine_size = None

    # Menentukan harga maksimal dengan regex yang lebih fleksibel (pastikan hanya menangkap angka setelah "under")
    match_price = re.search(r'under\s*(\d+)', input_text.lower())  # hanya menangkap angka setelah "under"
    if match_price:
        try:
            max_price = int(match_price.group(1))
        except (IndexError, TypeError):
            max_price = None

    # Memastikan angka lebih dari 3 digit dianggap sebagai harga
    numbers = re.findall(r'\d+', input_text)
    for number in numbers:
        if int(number) > 999:
            max_price = int(number)
            break

    return transmisi, engine_size, max_price

# Fungsi untuk memfilter mobil berdasarkan kriteria
def filter_cars(transmisi, engine_size, max_price):
    filtered_cars = df.copy()

    # Filter berdasarkan transmisi
    if transmisi:
        filtered_cars = filtered_cars[filtered_cars['transmisi'].str.contains(transmisi, case=False, na=False)]

    # Filter berdasarkan ukuran mesin
    if engine_size:
        filtered_cars = filtered_cars[filtered_cars['ukuran_mesin'] == engine_size]

    # Filter berdasarkan harga maksimal
    if max_price:
        filtered_cars = filtered_cars[filtered_cars['harga'] <= max_price]

    return filtered_cars

# Flask App Setup
app = Flask(__name__)
CORS(app)

# API endpoint to handle car preferences
@app.route('/api/get-cars', methods=['POST'])
def get_cars():
    data = request.get_json()  # Get the JSON data from the request
    user_input = data.get('query', '')

    # Extract criteria from the user's input
    transmisi, engine_size, max_price = extract_criteria(user_input)

    # Filter the cars based on the extracted criteria
    filtered_cars = filter_cars(transmisi, engine_size, max_price)

    if not filtered_cars.empty:
        # Prepare the response with car details
        cars_list = []
        for _, row in filtered_cars.iterrows():
            car_details = {
                'model': row['model'],
                'tahun': row['tahun'],
                'harga': row['harga'],
                'transmisi': row['transmisi'],
                'jarak_tempuh': row['jarak_tempuh'],
                'bahan_bakar': row['bahan_bakar'],
                'pajak': row['pajak'],
                'mpg': row['mpg'],
                'ukuran_mesin': row['ukuran_mesin']
            }
            cars_list.append(car_details)
        
        return jsonify({'cars': cars_list})
    else:
        return jsonify({'message': 'No cars found with the given criteria.'})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
