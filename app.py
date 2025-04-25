from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS  # Add this at the top

app = Flask(__name__)
CORS(app)  # Add this line right after creating your Flask app

model = joblib.load(r"Baisc_air_pollution_model.pkl")

traffic_map = {"Low": 0, "Medium": 1, "High": 2}
indus_map = {"None": 0, "Low": 1, "Medium": 2, "High": 3}
# Now, mapping to both pollution level and AQI range
pollution_map = {
    0: ("Low", "0-50"),
    1: ("Moderate", "51-150"),
    2: ("High", "151+")
}

def modelResponse(user_input):
    raw = user_input["data"]
    traffic = traffic_map.get(raw[0], 0)
    indus = indus_map.get(raw[1], 0)
    temp = float(raw[2])
    humidity = float(raw[3])
    tree = float(raw[4])
    input_data = [[traffic, indus, temp, humidity, tree]]
    prediction = model.predict(input_data)[0]
    pollution_level, aqi_range = pollution_map.get(prediction, ("Unknown", "N/A"))
    return pollution_level, aqi_range

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # Change "data" to "inputs" to match Framer code
        pollution_level, aqi_range = modelResponse({"data": [
            data["input1"],  # Traffic level
            data["input2"],  # Industrial activity
            data["input3"],  # Temperature
            data["input4"],  # Humidity
            data["input5"]   # Tree Density
        ]})
        return jsonify({
            "prediction": pollution_level.lower(),  # Ensure lowercase to match your Framer code
            "aqi": aqi_range.split("-")[0]  # Returns first number of range (e.g., "51" from "51-150")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
