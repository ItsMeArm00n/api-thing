from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load(r"Baisc_air_pollution_model.pkl")

traffic_map = {"Low": 0, "Medium": 1, "High": 2}
indus_map = {"None": 0, "Low": 1, "Medium": 2, "High": 3}
pollution_map = {0: "Low", 1: "Moderate", 2: "High"}

def modelResponse(user_input):
    raw = user_input["data"]
    traffic = traffic_map.get(raw[0], 0)
    indus = indus_map.get(raw[1], 0)
    temp = float(raw[2])
    humidity = float(raw[3])
    tree = float(raw[4])
    input_data = [[traffic, indus, temp, humidity, tree]]
    prediction = model.predict(input_data)[0]
    return pollution_map.get(prediction, "Unknown")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        result = modelResponse(data)
        return jsonify({"Pollution_Level": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
