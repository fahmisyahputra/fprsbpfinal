from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        # Extract data from form
        fuel = data.get('fuel')
        brand = data.get('brand')
        year = int(data.get('year'))
        price = float(data.get('price').replace('.', '').replace(',', ''))  # Remove formatting
        mileage = float(data.get('mileage').replace('.', '').replace(',', ''))  # Remove formatting
        engine = float(data.get('engine').replace('.', '').replace(',', ''))  # Remove formatting
        transmission = data.get('transmission')
        kilometers = float(data.get('kilometers'))

        # Dummy prediction logic
        predicted_value = (price * 0.8) - (mileage * 10) + (engine * 1000)

        return jsonify({'predicted_value': f"Rp {predicted_value:,.2f}"})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)