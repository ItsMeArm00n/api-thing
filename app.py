from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = joblib.load(r"Baisc_air_pollution_model.pkl")

traffic_map = {"Low": 0, "Medium": 1, "High": 2}
indus_map = {"Low": 1, "Medium": 2, "High": 3}
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
    pollution_level, _ = pollution_map.get(prediction, ("Unknown", "N/A"))
    return pollution_level

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # Now correctly checking inputs
        pollution_level = modelResponse({"data": [
            data.get("input1"),
            data.get("input2"),
            data.get("input3"),
            data.get("input4"),
            data.get("input5")
        ]})
        return jsonify({
            "prediction": pollution_level.lower()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
