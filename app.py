from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model Logistic Regression dan Label Encoder
model = joblib.load('model_log_regression.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil input dari form untuk semua fitur
        ph = float(request.form['ph'])
        temperature = float(request.form['temperature'])
        taste = int(request.form['taste'])
        odor = int(request.form['odor'])
        fat = int(request.form['fat'])
        turbidity = int(request.form['turbidity'])
        colour = int(request.form['colour'])

        # Format input ke dalam array sesuai fitur model
        features = np.array([[ph, temperature, taste, odor, fat, turbidity, colour]])

        # Prediksi menggunakan model
        prediction = model.predict(features)[0]

        # Konversi prediksi angka ke label kelas
        prediction_label = label_encoder.inverse_transform([prediction])[0]

        # Tampilkan hasil prediksi
        return jsonify({'prediction': prediction_label})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(debug=True, port=5001)
