import pickle
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.pipeline import Pipeline

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load dataset
df = pd.read_csv("dataset.csv")

# Load pre-trained KNN model
filename = "knn_model.pkl"
with open(filename, "rb") as file:
    loaded_knn_model = pickle.load(file)

# Recommendation function
def recommend_car_by_user_input(user_input, df, pipeline, n_recommendations=5):
    # Fill missing categorical values with the most common value
    user_input = user_input.copy()
    for col in ["Merek", "Model"]:
        if not user_input[col]:
            user_input[col] = df[col].mode()[0]

    # Create a DataFrame for the user's preferences
    user_input_df = pd.DataFrame([user_input])

    # Preprocess the user's input
    preprocessed_input = pipeline.named_steps["preprocessor"].transform(user_input_df)

    # Find nearest neighbors
    distances, indices = pipeline.named_steps["knn"].kneighbors(preprocessed_input, n_neighbors=n_recommendations)

    # Retrieve the recommended cars
    recommended_cars = df.iloc[indices[0]]
    return recommended_cars


# API endpoint to handle car recommendations
@app.route("/api/get-cars", methods=["POST"])
def get_cars():
    try:
        # Parse input JSON
        data = request.get_json()
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

        # Get recommendations
        recommendations = recommend_car_by_user_input(user_preferences, df, loaded_knn_model)

        # Format response
        results = recommendations.to_dict(orient="records")
        return jsonify({"cars": results}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
