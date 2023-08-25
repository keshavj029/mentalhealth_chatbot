from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load("Stress identification NLP")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']

    # Make predictions using the loaded model
    prediction = model.predict([text])[0]

    response = {
        'text': text,
        'stress_prediction': prediction
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
